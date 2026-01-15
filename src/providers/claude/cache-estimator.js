/**
 * Kiro 缓存推测模块
 * 用于估算 Claude Prompt Caching 的缓存命中情况
 */

import * as crypto from 'crypto';
import { countTokens } from '@anthropic-ai/tokenizer';

/**
 * Kiro 缓存推测配置
 *
 * OPTIMISTIC_MATCHING (默认: true):
 *   启用时，允许在消息序列中出现"空洞"的缓存命中。
 *   示例：如果消息 [1,2,3,4,5] 存在，消息 3 改变但 4-5 与历史匹配，
 *   则 [1,2] 和 [4,5] 都会被计为 cache_read。
 *
 *   ⚠️ 警告：这不反映 Claude 的实际缓存行为。
 *   Claude 使用严格的前缀匹配 - 一旦前缀断裂，所有后续消息
 *   都会变成 cache_creation，无论内容是否匹配。
 *
 *   此模式仅用于乐观估算。实际账单会更高。
 *
 *   禁用方式: KIRO_OPTIMISTIC_CACHE=false
 */
// 缓存推测配置
export const CACHE_ESTIMATION_CONFIG = {
    HISTORY_TTL: 300000,            // 5分钟（Claude 默认缓存 TTL）
    MAX_CACHE_SIZE: 500,            // LRU 缓存最大条目数
    CONFIDENCE_THRESHOLD: 20,       // 低于此值不应用推测（百分比）
    MIN_MATCH_RATIO: 0.1,           // 最小匹配比例 10%
    MIN_MATCHED_TOKENS: 1024,       // 最小匹配 token 数
    ENABLED: true,                  // 是否启用缓存推测
    // tool_result 处理策略: 'strict'(严格匹配), 'ignore'(忽略变化), 'name_only'(只匹配工具名)
    TOOL_RESULT_STRATEGY: 'strict',
    DEBUG: process.env.KIRO_CACHE_DEBUG === 'true',  // 调试日志开关
    // 乐观匹配模式：允许跳过中间不匹配的消息继续匹配后续消息（默认开启）
    // 注意：这不反映 Claude 实际缓存行为，仅用于乐观估算
    OPTIMISTIC_MATCHING: process.env.KIRO_OPTIMISTIC_CACHE !== 'false',
};

// 全局日志级别控制
const LOG_LEVEL = {
    DEBUG: process.env.KIRO_LOG_LEVEL === 'debug',
    INFO: process.env.KIRO_LOG_LEVEL !== 'error' && process.env.KIRO_LOG_LEVEL !== 'warn',
    WARN: process.env.KIRO_LOG_LEVEL !== 'error',
    ERROR: true,
};

// 日志工具函数
const logger = {
    debug: (...args) => LOG_LEVEL.DEBUG && console.log(...args),
    info: (...args) => LOG_LEVEL.INFO && console.log(...args),
    warn: (...args) => LOG_LEVEL.WARN && console.warn(...args),
    error: (...args) => LOG_LEVEL.ERROR && console.error(...args),
};

// 模型最小缓存阈值
export const MODEL_MIN_CACHE_TOKENS = {
    'claude-opus-4-5': 4096,
    'claude-opus-4': 1024,
    'claude-sonnet-4-5': 1024,
    'claude-sonnet-4': 1024,
    'claude-sonnet-3-7': 1024,
    'claude-haiku-4-5': 4096,
    'claude-haiku-3-5': 2048,
    'claude-haiku-3': 2048,
    'default': 1024
};

// 块类型定义（保留用于未来扩展）
// const BLOCK_TYPES = {
//     TOOLS: 'tools',
//     TOOL_CHOICE: 'tool_choice',
//     THINKING: 'thinking',
//     SYSTEM: 'system',
//     MESSAGE: 'message'
// };

/**
 * 获取模型的最小缓存 token 阈值
 * @param {string} model - 模型名称
 * @returns {number} 最小缓存 token 数
 */
export function getMinCacheTokens(model) {
    if (!model) return MODEL_MIN_CACHE_TOKENS.default;

    // 精确匹配
    if (MODEL_MIN_CACHE_TOKENS[model]) {
        return MODEL_MIN_CACHE_TOKENS[model];
    }

    // 前缀匹配
    for (const [prefix, tokens] of Object.entries(MODEL_MIN_CACHE_TOKENS)) {
        if (prefix !== 'default' && model.startsWith(prefix)) {
            return tokens;
        }
    }

    return MODEL_MIN_CACHE_TOKENS.default;
}

// 以下函数暂未使用，保留用于未来扩展
// /**
//  * FNV-1a 哈希函数（32位）
//  * @param {string} str - 要哈希的字符串
//  * @returns {string} 8字符十六进制哈希
//  */
// function fnv1aHash(str) {
//     let hash = 2166136261; // FNV offset basis
//     for (let i = 0; i < str.length; i++) {
//         hash ^= str.charCodeAt(i);
//         hash = Math.imul(hash, 16777619); // FNV prime
//         hash = hash >>> 0; // 保持32位无符号
//     }
//     return hash.toString(16).padStart(8, '0');
// }

// /**
//  * 稳定 JSON 序列化（key 排序）
//  * @param {any} obj - 要序列化的对象
//  * @returns {string} 稳定的 JSON 字符串
//  */
// function stableStringify(obj) {
//     if (obj === null || obj === undefined) return '';
//     if (typeof obj !== 'object') return String(obj);
//
//     if (Array.isArray(obj)) {
//         return '[' + obj.map(item => stableStringify(item)).join(',') + ']';
//     }
//
//     const keys = Object.keys(obj).sort();
//     const parts = keys.map(key => {
//         const value = obj[key];
//         return JSON.stringify(key) + ':' + (
//             typeof value === 'object' && value !== null
//                 ? stableStringify(value)
//                 : JSON.stringify(value)
//         );
//     });
//     return '{' + parts.join(',') + '}';
// }

/**
 * 获取图片指纹（用于替代完整 base64）
 * @param {string} base64 - base64 编码的图片数据
 * @returns {string} 图片指纹
 */
export function getImageFingerprint(base64) {
    if (!base64 || typeof base64 !== 'string') return '';
    const len = base64.length;
    // 长度 + 首尾各 32 字符
    const head = base64.substring(0, 32);
    const tail = base64.substring(Math.max(0, len - 32));
    return `img:${len}:${head}:${tail}`;
}

// 以下函数暂未使用，保留用于未来扩展
// /**
//  * 规范化内容（移除动态字段）
//  * @param {any} content - 原始内容
//  * @returns {any} 规范化后的内容
//  */
// function normalizeContent(content) {
//     if (!content) return null;
//     if (typeof content === 'string') return content;
//
//     if (Array.isArray(content)) {
//         return content.map(item => normalizeContent(item));
//     }
//
//     if (typeof content === 'object') {
//         const normalized = {};
//         for (const [key, value] of Object.entries(content)) {
//             // 移除动态字段
//             if (['cache_control', 'id', 'tool_use_id', 'signature'].includes(key)) {
//                 continue;
//             }
//             // 图片数据用指纹替代
//             if (key === 'data' && content.type === 'image') {
//                 normalized[key] = getImageFingerprint(value);
//             } else if (typeof value === 'object' && value !== null) {
//                 normalized[key] = normalizeContent(value);
//             } else {
//                 normalized[key] = value;
//             }
//         }
//         return normalized;
//     }
//
//     return content;
// }

// /**
//  * 检查内容是否包含 cache_control
//  * @param {any} content - 内容
//  * @returns {boolean}
//  */
// function contentHasCacheControl(content) {
//     if (!content) return false;
//
//     if (typeof content === 'object' && content.cache_control) {
//         return true;
//     }
//
//     if (Array.isArray(content)) {
//         return content.some(item => contentHasCacheControl(item));
//     }
//
//     return false;
// }

/**
 * 简单的 LRU 缓存实现
 *
 * TTL 行为：滑动窗口（与 Claude API 一致）
 * - Claude 官方文档："The cache is refreshed for no additional cost each time the cached content is used"
 * - 每次访问都会重置 5 分钟倒计时
 * - 只要持续使用，缓存就不会过期
 */
export class SimpleLRUCache {
    constructor(maxSize = 500, ttl = 300000) {
        this.maxSize = maxSize;
        this.ttl = ttl;
        this.cache = new Map();
    }

    get(key) {
        const item = this.cache.get(key);
        if (!item) return null;

        // 检查是否过期
        const now = Date.now();
        if (now - item.timestamp > this.ttl) {
            this.cache.delete(key);
            return null;
        }

        // 滑动 TTL: 更新时间戳，重置过期时间（与 Claude 行为一致）
        item.timestamp = now;
        return item;
    }

    set(key, value) {
        // 如果已存在，直接更新（避免删除再插入）
        const existing = this.cache.get(key);
        if (existing) {
            Object.assign(existing, value);
            existing.timestamp = Date.now();
            return;
        }

        // 检查容量
        if (this.cache.size >= this.maxSize) {
            // 删除最旧的条目
            const firstKey = this.cache.keys().next().value;
            this.cache.delete(firstKey);
        }

        this.cache.set(key, {
            ...value,
            timestamp: Date.now()
        });
    }

    has(key) {
        // 不调用 get() 以避免改变 LRU 顺序
        const item = this.cache.get(key);
        if (!item) return false;
        // 检查是否过期
        if (Date.now() - item.timestamp > this.ttl) {
            this.cache.delete(key);
            return false;
        }
        return true;
    }

    getValue(key) {
        const item = this.get(key);
        return item ? item.value : null;
    }

    setValue(key, value) {
        this.set(key, { value });
    }

    get size() {
        return this.cache.size;
    }

    clear() {
        this.cache.clear();
    }
}

/**
 * Kiro 缓存推测器（块级哈希匹配版本）
 * 用于推测 Prompt Caching 的缓存命中情况
 *
 * 设计原则：
 * 1. 块级序列化：按 Claude 顺序（tools → tool_choice → thinking → system → messages）
 * 2. FNV-1a 哈希：每个块独立计算哈希
 * 3. 前缀匹配：逐块比对找最长匹配前缀
 * 4. 置信度计算：基于匹配比例 + token 数量加成 + 时间衰减
 */
export class KiroCacheEstimator {
    constructor(config = {}) {
        this.config = {
            historyTTL: config.historyTTL || CACHE_ESTIMATION_CONFIG.HISTORY_TTL,
            maxCacheSize: config.maxCacheSize || CACHE_ESTIMATION_CONFIG.MAX_CACHE_SIZE,
            confidenceThreshold: config.confidenceThreshold || CACHE_ESTIMATION_CONFIG.CONFIDENCE_THRESHOLD,
            minMatchRatio: config.minMatchRatio || CACHE_ESTIMATION_CONFIG.MIN_MATCH_RATIO,
            minMatchedTokens: config.minMatchedTokens || CACHE_ESTIMATION_CONFIG.MIN_MATCHED_TOKENS,
            minCacheableTokens: config.minCacheableTokens || CACHE_ESTIMATION_CONFIG.MIN_MATCHED_TOKENS,
            enabled: config.enabled !== false && CACHE_ESTIMATION_CONFIG.ENABLED,
            toolResultStrategy: config.toolResultStrategy || CACHE_ESTIMATION_CONFIG.TOOL_RESULT_STRATEGY
        };

        // 请求缓存 (key: prefixHash, value: cached prefix info)，带 TTL 自动过期
        this.requestCache = new SimpleLRUCache(
            this.config.maxCacheSize,
            this.config.historyTTL  // 使用 5 分钟 TTL，与 Claude 缓存过期时间一致
        );

        // 统计信息
        this.stats = {
            totalRequests: 0,
            estimatedCacheHits: 0,
            estimatedCacheCreations: 0,
            skippedLowConfidence: 0,
            skippedNoCacheControl: 0,
            skippedBelowThreshold: 0
        };
    }

    /**
     * 主入口：估算缓存 token
     * @param {Object} request - 原始请求
     * @param {number} totalInputTokens - 总输入 token 数（包含所有内容）
     * @param {string} model - 模型名称
     * @returns {Object} { cache_read_input_tokens, cache_creation_input_tokens, uncached_input_tokens, _estimation }
     *
     * Claude API 计费公式：
     * total_input = cache_read_input_tokens + cache_creation_input_tokens + uncached_input_tokens
     * - cache_read: 从缓存读取的 token (0.1x 价格)
     * - cache_creation: 写入缓存的 token (1.25x 价格)
     * - uncached_input: 不参与缓存的普通 token (1x 价格)
     */
    estimateCacheTokens(request, totalInputTokens, model) {
        if (!this.config.enabled) {
            return {
                cache_read_input_tokens: 0,
                cache_creation_input_tokens: 0,
                uncached_input_tokens: totalInputTokens,
                _estimation: { estimated: false, source: 'disabled' }
            };
        }

        this.stats.totalRequests++;

        // 1. 提取可缓存内容（区分静态前缀和缓存消息，每个消息带独立哈希）
        const cacheableContent = this.extractCacheableContent(request);

        // 2. 如果没有任何 cache_control，直接返回全部作为 uncached
        if (!cacheableContent.hasCacheControl) {
            this.stats.skippedNoCacheControl++;
            return {
                cache_read_input_tokens: 0,
                cache_creation_input_tokens: 0,
                uncached_input_tokens: totalInputTokens,
                _estimation: {
                    estimated: true,
                    source: 'no_cache_control',
                    confidence: 1.0,
                    totalInputTokens
                }
            };
        }

        // 3. 计算静态前缀的 token 数（system + tools）
        const staticPrefixTokens = this.countStaticPrefixTokens(cacheableContent);

        // 4. 获取从头到 lastCacheBreakpoint 的所有消息 tokens
        const prefixMessagesTokens = cacheableContent.prefixMessagesTokens;

        // 5. 检查 system 和 tools 是否有 cache_control
        const systemHasCacheControl = cacheableContent.staticPrefix.systemHasCacheControl;
        const toolsHasCacheControl = cacheableContent.staticPrefix.toolsHasCacheControl;

        // 6. 总可缓存 token 计算
        // 静态前缀只有在 system 或 tools 有 cache_control 时才计入缓存
        const staticCacheable = (systemHasCacheControl || toolsHasCacheControl) ? staticPrefixTokens : 0;
        const totalCacheableTokens = staticCacheable + prefixMessagesTokens;

        // 7. 计算不参与缓存的 token（断点之后的消息）
        // uncached = total - cacheable
        const uncachedTokens = Math.max(0, totalInputTokens - totalCacheableTokens);

        // 8. 检查是否满足最小缓存条件
        if (totalCacheableTokens < this.config.minCacheableTokens) {
            this.stats.skippedBelowThreshold++;
            return {
                cache_read_input_tokens: 0,
                cache_creation_input_tokens: 0,
                uncached_input_tokens: totalInputTokens,  // 全部作为普通输入
                _estimation: {
                    estimated: true,
                    source: 'below_threshold',
                    confidence: 0.95,
                    staticPrefixTokens,
                    staticCacheable,
                    prefixMessagesTokens,
                    totalCacheableTokens,
                    uncachedTokens,
                    totalInputTokens,
                    systemHasCacheControl,
                    toolsHasCacheControl,
                    lastCacheBreakpoint: cacheableContent.lastCacheBreakpoint,
                    minRequired: this.config.minCacheableTokens
                }
            };
        }

        // 9. 计算静态前缀哈希（system + tools + tool_choice + thinking）
        const prefixHash = this.computeContentHash(cacheableContent, model, request);

        // 10. 检查静态前缀是否命中
        const cachedPrefixItem = this.requestCache.get(prefixHash);
        const cachedPrefix = cachedPrefixItem ? cachedPrefixItem.value : null;

        if (cachedPrefix) {
            // 静态前缀命中，逐个比较缓存前缀内的消息
            const matchDetails = [];
            const previousMessages = cachedPrefix.cachedMessages || [];
            const currentMessages = cacheableContent.cachedMessages;
            const allMessagesTokens = cacheableContent.allMessagesTokens;
            const lastCacheBreakpoint = cacheableContent.lastCacheBreakpoint;

            // Claude 缓存是严格前缀匹配，一旦某个位置断开，后续全部无法命中
            let prefixBroken = false;
            let lastMatchedBreakpoint = -1;  // 最后一个匹配的消息索引

            // 逐个检查缓存前缀内的每个消息
            if (CACHE_ESTIMATION_CONFIG.DEBUG) {
                console.log(`[Kiro Cache Debug] prefixHash=${prefixHash}, currentMessages=${currentMessages.length}, previousMessages=${previousMessages.length}`);
            }
            for (let i = 0; i < currentMessages.length; i++) {
                const currentMsg = currentMessages[i];
                const previousMsg = previousMessages[i];

                // 检查内容是否匹配
                const contentMatches = previousMsg &&
                    previousMsg.index === currentMsg.index &&
                    previousMsg.contentHash === currentMsg.contentHash;

                if (CACHE_ESTIMATION_CONFIG.OPTIMISTIC_MATCHING) {
                    // 乐观模式：即使前面有不匹配，也继续检查当前消息
                    // 调试日志：输出每条消息的哈希比较结果
                    if (CACHE_ESTIMATION_CONFIG.DEBUG) {
                        console.log(`[Kiro Cache Debug] [OPTIMISTIC] Message ${i}: ` +
                            `index=${currentMsg.index}, ` +
                            `currentHash=${currentMsg.contentHash}, ` +
                            `previousHash=${previousMsg?.contentHash || 'N/A'}, ` +
                            `tokens=${currentMsg.tokens}, ` +
                            `match=${contentMatches}`);

                        // 当哈希不匹配时，输出详细内容用于调试
                        if (!contentMatches && previousMsg && previousMsg.contentHash !== currentMsg.contentHash) {
                            console.log(`[Kiro Cache Debug] ==={i} HASH MISMATCH ===`);
                            console.log(`[Kiro Cache Debug]   Role: ${currentMsg.role}`);
                            console.log(`[Kiro Cache Debug]   Content Type: ${currentMsg._debug?.contentType || 'unknown'}`);
                            console.log(`[Kiro Cache Debug]   Current hashInput length: ${currentMsg._debug?.hashInputLength || 'N/A'}`);
                            console.log(`[Kiro Cache Debug]   Previous hashInput length: ${previousMsg._debug?.hashInputLength || 'N/A'}`);
                            console.log(`[Kiro Cache Debug]   Current preview: ${currentMsg._debug?.hashInputPreview || 'N/A'}`);
                            console.log(`[Kiro Cache Debug]   Previous preview: ${previousMsg._debug?.hashInputPreview || 'N/A'}`);
                            console.log(`[Kiro Cache Debug] === END MISMATCH ===`);
                        }
                    }

                    if (contentMatches) {
                        // 内容匹配，标记为 cache_read
                        lastMatchedBreakpoint = currentMsg.index;
                        matchDetails.push({
                            index: currentMsg.index,
                            status: 'hit',
                            tokens: currentMsg.tokens
                        });
                    } else {
                        // 内容不匹配，标记为 cache_creation
                        matchDetails.push({
                            index: currentMsg.index,
                            status: previousMsg ? 'changed' : 'new',
                            tokens: currentMsg.tokens
                        });
                    }
                } else {
                    // 严格前缀匹配模式（原有逻辑）
                    const isMatch = !prefixBroken && contentMatches;

                    // 调试日志：输出每条消息的哈希比较结果
                    if (CACHE_ESTIMATION_CONFIG.DEBUG) {
                        console.log(`[Kiro Cache Debug] [STRICT] Message ${i}: ` +
                            `index=${currentMsg.index}, ` +
                            `currentHash=${currentMsg.contentHash}, ` +
                            `previousHash=${previousMsg?.contentHash || 'N/A'}, ` +
                            `tokens=${currentMsg.tokens}, ` +
                            `match=${isMatch}, ` +
                            `prefixBroken=${prefixBroken}`);

                        // 当哈希不匹配时，输出详细内容用于调试
                        if (!isMatch && previousMsg && previousMsg.contentHash !== currentMsg.contentHash) {
                            console.log(`[Kiro Cache Debug] === Message ${i} HASH MISMATCH ===`);
                            console.log(`[Kiro Cache Debug]   Role: ${currentMsg.role}`);
                            console.log(`[Kiro Cache Debug]   Content Type: ${currentMsg._debug?.contentType || 'unknown'}`);
                            console.log(`[Kiro Cache Debug]   Current hashInput length: ${currentMsg._debug?.hashInputLength || 'N/A'}`);
                            console.log(`[Kiro Cache Debug]   Previous hashInput length: ${previousMsg._debug?.hashInputLength || 'N/A'}`);
                            console.log(`[Kiro Cache Debug]   Current preview: ${currentMsg._debug?.hashInputPreview || 'N/A'}`);
                            console.log(`[Kiro Cache Debug]   Previous preview: ${previousMsg._debug?.hashInputPreview || 'N/A'}`);
                            console.log(`[Kiro Cache Debug] === END MISMATCH ===`);
                        }
                    }

                    if (isMatch) {
                        // 内容完全匹配且前缀未断
                        lastMatchedBreakpoint = currentMsg.index;
                        matchDetails.push({
                            index: currentMsg.index,
                            status: 'hit',
                            tokens: currentMsg.tokens
                        });
                    } else {
                        // 前缀断开，当前及后续全部是 cache_creation
                        prefixBroken = true;
                        matchDetails.push({
                            index: currentMsg.index,
                            status: previousMsg ? 'changed' : 'new',
                            tokens: currentMsg.tokens
                        });
                    }
                }
            }
            if (CACHE_ESTIMATION_CONFIG.DEBUG) {
                const mode = CACHE_ESTIMATION_CONFIG.OPTIMISTIC_MATCHING ? 'OPTIMISTIC' : 'STRICT';
                console.log(`[Kiro Cache Debug] Mode: ${mode}`);
                console.log(`[Kiro Cache Debug] Result: lastMatchedBreakpoint=${lastMatchedBreakpoint}, prefixBroken=${prefixBroken}, staticCacheable=${staticCacheable}`);
                console.log(`[Kiro Cache Debug] Match details: ${matchDetails.map(m => `[${m.index}] ${m.status} (${m.tokens} tokens)`).join(', ')}`);
            }

            // 计算 cache_read 和 cache_creation
            // 静态前缀只有在 system 或 tools 有 cache_control 时才计入
            let cacheRead = staticCacheable;
            let cacheCreation = 0;

            if (CACHE_ESTIMATION_CONFIG.OPTIMISTIC_MATCHING) {
                // 乐观模式：根据每条消息的匹配状态分别计算
                for (const detail of matchDetails) {
                    if (detail.status === 'hit') {
                        cacheRead += detail.tokens;
                    } else {
                        // 'changed' 或 'new' 状态的消息计入 cache_creation
                        cacheCreation += detail.tokens;
                    }
                }
            } else if (currentMessages.length === 0) {
                // 没有缓存前缀消息（lastCacheBreakpoint < 0），只有静态前缀
                // 静态前缀命中，全部作为 cache_read
                cacheCreation = 0;
            } else if (!prefixBroken) {
                // 所有消息都匹配，全部作为 cache_read
                for (let i = 0; i <= lastCacheBreakpoint && i < allMessagesTokens.length; i++) {
                    cacheRead += allMessagesTokens[i] || 0;
                }
                cacheCreation = 0;
            } else {
                // 部分匹配
                if (lastMatchedBreakpoint >= 0) {
                    for (let i = 0; i <= lastMatchedBreakpoint && i < allMessagesTokens.length; i++) {
                        cacheRead += allMessagesTokens[i] || 0;
                    }
                }
                // 从 lastMatchedBreakpoint+1 到 lastCacheBreakpoint 的消息作为 cache_creation
                for (let i = lastMatchedBreakpoint + 1; i <= lastCacheBreakpoint && i < allMessagesTokens.length; i++) {
                    cacheCreation += allMessagesTokens[i] || 0;
                }
            }

            // 更新缓存记录
            this.requestCache.set(prefixHash, {
                value: {
                    staticPrefixTokens,
                    staticCacheable,
                    prefixMessagesTokens,
                    lastCacheBreakpoint,
                    cachedMessages: currentMessages.map(m => ({
                        index: m.index,
                        contentHash: m.contentHash,
                        tokens: m.tokens,
                        role: m.role
                    })),
                    allMessagesTokens: [...allMessagesTokens],
                    hitCount: (cachedPrefix.hitCount || 0) + 1
                }
            });

            const hasNewOrChanged = cacheCreation > 0;
            const source = hasNewOrChanged ? 'cache_hit_partial' : 'cache_hit';
            this.stats.estimatedCacheHits++;

            return {
                cache_read_input_tokens: cacheRead,
                cache_creation_input_tokens: cacheCreation,
                uncached_input_tokens: uncachedTokens,
                _estimation: {
                    estimated: true,
                    source,
                    confidence: 0.85,
                    optimistic: CACHE_ESTIMATION_CONFIG.OPTIMISTIC_MATCHING,
                    staticPrefixTokens,
                    staticCacheable,
                    prefixMessagesTokens,
                    totalCacheableTokens,
                    uncachedTokens,
                    totalInputTokens,
                    systemHasCacheControl,
                    toolsHasCacheControl,
                    lastCacheBreakpoint,
                    lastMatchedBreakpoint,
                    hitCount: cachedPrefix.hitCount + 1,
                    matchDetails,
                    contentHash: prefixHash
                }
            };
        } else {
            // 静态前缀未命中，全部作为缓存创建
            this.stats.estimatedCacheCreations++;

            // 存储每个消息的独立信息
            this.requestCache.set(prefixHash, {
                value: {
                    staticPrefixTokens,
                    staticCacheable,
                    prefixMessagesTokens,
                    lastCacheBreakpoint: cacheableContent.lastCacheBreakpoint,
                    cachedMessages: cacheableContent.cachedMessages.map(m => ({
                        index: m.index,
                        contentHash: m.contentHash,
                        tokens: m.tokens,
                        role: m.role
                    })),
                    allMessagesTokens: [...cacheableContent.allMessagesTokens],
                    hitCount: 0
                }
            });

            return {
                cache_read_input_tokens: 0,
                cache_creation_input_tokens: totalCacheableTokens,
                uncached_input_tokens: uncachedTokens,
                _estimation: {
                    estimated: true,
                    source: 'cache_creation',
                    confidence: 0.90,
                    staticPrefixTokens,
                    staticCacheable,
                    prefixMessagesTokens,
                    totalCacheableTokens,
                    uncachedTokens,
                    totalInputTokens,
                    systemHasCacheControl,
                    toolsHasCacheControl,
                    lastCacheBreakpoint: cacheableContent.lastCacheBreakpoint,
                    contentHash: prefixHash
                }
            };
        }
    }

    /**
     * 提取可缓存内容
     * 区分静态前缀（system + tools）和带缓存断点的消息
     * 为每个缓存消息计算独立的哈希和 token 数
     *
     * 缓存推测策略：
     * - 只有显式设置 cache_control 时才会创建/读取缓存
     * - Claude 缓存是前缀匹配的，会缓存从头到 cache_control 标记位置（包含该位置）
     * - 如果没有任何 cache_control，则不产生缓存
     */
    extractCacheableContent(request) {
        const cacheable = {
            // 静态前缀（用于判断缓存命中）
            staticPrefix: {
                system: null,
                systemHasCacheControl: false,
                toolsHasCacheControl: false,
                tools: null
            },
            // 是否有任何 cache_control 标记
            hasCacheControl: false,
            // 缓存前缀内的消息（用于哈希比较，检测内容变化）
            cachedMessages: [],
            // 缓存前缀的最后一个消息索引（包含 cache_control 的消息）
            lastCacheBreakpoint: -1,
            // 从头到 lastCacheBreakpoint 的所有消息 tokens
            prefixMessagesTokens: 0,
            // 每个消息的 token 数（用于精确计算 cache_read/cache_creation）
            allMessagesTokens: []
        };

        // 1. System prompt
        if (request.system) {
            cacheable.staticPrefix.system = request.system;
            // 检查是否有显式的 cache_control
            if (Array.isArray(request.system)) {
                cacheable.staticPrefix.systemHasCacheControl = request.system.some(p => p && p.cache_control);
            } else if (typeof request.system === 'object' && request.system !== null && request.system.cache_control) {
                cacheable.staticPrefix.systemHasCacheControl = true;
            }
            if (cacheable.staticPrefix.systemHasCacheControl) {
                cacheable.hasCacheControl = true;
            }
        }

        // 2. Tools 定义 - 检查是否有 cache_control
        if (request.tools && request.tools.length > 0) {
            cacheable.staticPrefix.tools = request.tools;
            // 检查 tools 数组最后一个元素是否有 cache_control
            const lastTool = request.tools[request.tools.length - 1];
            if (lastTool && lastTool.cache_control) {
                cacheable.staticPrefix.toolsHasCacheControl = true;
                cacheable.hasCacheControl = true;
            }
        }

        // 3. 遍历 messages，计算每个消息的 tokens，找到最后一个 cache_control 的位置
        if (request.messages && request.messages.length > 0) {
            // 查找最后一个带 cache_control 的消息索引
            // Claude 缓存包含 cache_control 标记的消息本身
            let lastCacheControlIndex = -1;
            for (let i = 0; i < request.messages.length; i++) {
                const msg = request.messages[i];
                if (this.messageHasCacheControl(msg)) {
                    lastCacheControlIndex = i;
                    cacheable.hasCacheControl = true;
                }
            }

            // 计算每个消息的 token 数
            for (let i = 0; i < request.messages.length; i++) {
                const msg = request.messages[i];
                const contentText = this.contentToText(msg.content, msg.role);
                const msgTokens = this.countTokensSimple(contentText);
                cacheable.allMessagesTokens.push(msgTokens);
            }

            // 确定缓存前缀的范围
            // Claude 缓存包含 cache_control 标记的消息本身
            cacheable.lastCacheBreakpoint = lastCacheControlIndex;

            // 为缓存前缀内的每个消息计算哈希
            if (cacheable.lastCacheBreakpoint >= 0) {
                for (let i = 0; i <= cacheable.lastCacheBreakpoint; i++) {
                    const msg = request.messages[i];
                    const hashInput = this.contentToTextForHash(msg.content, msg.role);
                    cacheable.cachedMessages.push({
                        index: i,
                        role: msg.role,
                        // 使用稳定字段计算哈希，排除易变的 tool_use_id/id/input
                        contentHash: this.simpleHash(hashInput),
                        tokens: cacheable.allMessagesTokens[i],
                        // 调试用：保存内容摘要（不存储完整内容以节省内存）
                        _debug: CACHE_ESTIMATION_CONFIG.DEBUG ? {
                            hashInputPreview: hashInput.substring(0, 200),
                            hashInputLength: hashInput.length,
                            contentType: Array.isArray(msg.content)
                                ? msg.content.map(c => c?.type || typeof c).join(',')
                                : typeof msg.content
                        } : null
                    });
                    cacheable.prefixMessagesTokens += cacheable.allMessagesTokens[i];
                }
            }
        }

        return cacheable;
    }

    /**
     * 检查消息是否包含 cache_control
     * 支持数组格式、对象格式和消息级 cache_control
     */
    messageHasCacheControl(msg) {
        if (!msg) return false;

        // 检查消息级 cache_control
        if (msg.cache_control) return true;

        // 检查 content 数组中的 cache_control
        if (Array.isArray(msg.content)) {
            return msg.content.some(p => p && p.cache_control);
        }

        // 检查 content 对象的 cache_control
        if (typeof msg.content === 'object' && msg.content !== null && msg.content.cache_control) {
            return true;
        }

        return false;
    }

    /**
     * 计算静态前缀的 token 数
     * system 和 tools 始终计入（即使没有 cache_control）
     */
    countStaticPrefixTokens(cacheableContent) {
        let tokens = 0;

        // system 始终计入缓存推测
        if (cacheableContent.staticPrefix.system) {
            const systemText = this.contentToText(cacheableContent.staticPrefix.system);
            tokens += this.countTokensSimple(systemText);
        }

        // tools 始终计入缓存推测（使用缓存的序列化结果）
        if (cacheableContent.staticPrefix.tools) {
            if (!cacheableContent._toolsJsonCache) {
                cacheableContent._toolsJsonCache = JSON.stringify(cacheableContent.staticPrefix.tools);
            }
            tokens += this.countTokensSimple(cacheableContent._toolsJsonCache);
        }

        return tokens;
    }

    /**
     * 计算内容哈希（只包含静态前缀：system + tools）
     * 使用稳定字段，排除动态内容
     */
    computeContentHash(cacheableContent, model, request = {}) {
        // 对 system 和 tools 使用稳定字段提取
        const stableSystem = this.extractStableFields(cacheableContent.staticPrefix.system);
        const stableTools = this.extractStableToolsFields(cacheableContent.staticPrefix.tools);

        // 缓存哈希输入字符串
        const hashInputObj = {
            model: model,
            system: stableSystem,
            tools: stableTools,
            // 添加影响缓存行为的关键字段
            tool_choice: request.tool_choice || null,
            thinking: request.thinking ? {
                type: request.thinking.type,
                budget_tokens: request.thinking.budget_tokens
            } : null
        };
        const hashInput = JSON.stringify(hashInputObj);

        return this.simpleHash(hashInput);
    }

    /**
     * 提取稳定字段（用于哈希计算）
     * 排除可能包含动态内容的字段
     */
    extractStableFields(content) {
        if (!content) return null;
        if (typeof content === 'string') return content;

        if (Array.isArray(content)) {
            return content.map(c => {
                if (typeof c === 'string') return c;
                if (typeof c === 'object' && c !== null) {
                    // 只保留稳定字段
                    const stable = {};
                    if (c.type) stable.type = c.type;
                    if (c.text) stable.text = c.text;
                    if (c.cache_control) stable.cache_control = c.cache_control;
                    return stable;
                }
                return c;
            });
        }

        if (typeof content === 'object') {
            const stable = {};
            if (content.type) stable.type = content.type;
            if (content.text) stable.text = content.text;
            if (content.cache_control) stable.cache_control = content.cache_control;
            return stable;
        }

        return content;
    }

    /**
     * 提取工具定义的稳定字段
     * 工具定义通常是稳定的，但排除可能的动态字段
     */
    extractStableToolsFields(tools) {
        if (!tools || !Array.isArray(tools)) return null;

        return tools.map(tool => {
            if (typeof tool !== 'object' || tool === null) return tool;
            // 工具定义的核心字段：name, description, input_schema
            const stable = {};
            if (tool.name) stable.name = tool.name;
            if (tool.description) stable.description = tool.description;
            if (tool.input_schema) stable.input_schema = tool.input_schema;
            return stable;
        });
    }

    /**
     * 内容转文本（包含元数据用于哈希计算和 token 估算）
     * @param {any} content - 消息内容
     * @param {string} role - 消息角色（可选）
     */
    contentToText(content, role = null) {
        let parts = [];
        if (role) parts.push(role);

        if (!content) return parts.join(' ');
        if (typeof content === 'string') {
            parts.push(content);
            return parts.join(' ');
        }

        if (Array.isArray(content)) {
            for (const c of content) {
                if (typeof c === 'string') {
                    parts.push(c);
                    continue;
                }
                // 包含元数据：type, cache_control, tool_use_id, name, id 等
                if (c.type) parts.push(c.type);
                if (c.cache_control) parts.push(JSON.stringify(c.cache_control));
                if (c.text) parts.push(c.text);
                if (c.thinking) parts.push(c.thinking);
                if (c.tool_use_id) parts.push(c.tool_use_id);
                if (c.name) parts.push(c.name);
                if (c.id) parts.push(c.id);
                if (c.input) parts.push(JSON.stringify(c.input));
                if (c.content) parts.push(this.contentToText(c.content));
            }
            return parts.join(' ');
        }

        parts.push(JSON.stringify(content));
        return parts.join(' ');
    }

    /**
     * 规范化文本中的特殊字符（用于哈希计算）
     * 将各种箭头字符统一为标准形式，避免编码差异导致哈希不匹配
     *
     * 性能优化：使用单个正则表达式和替换函数，替代原来的 13 次正则替换
     */
    normalizeTextForHash(text) {
        if (!text || typeof text !== 'string') return text;

        // 性能优化：使用单个正则匹配所有需要替换的模式
        return text.replace(
            /[→⇒➜➔➙➛►▶︎▸⮕]|[←⇐◄◀︎◂⬅]|[↔⇔]|[\uFFFD�]|[\u{E000}-\u{F8FF}]|[\x00-\x08\x0B\x0C\x0E-\x1F]|(->){2,}|(<-){2,}|(<->){2,}|\.{4,}|-{3,}|_{3,}|\s+$/gmu,
            (match) => {
                // 右箭头类
                if (/^[→⇒➜➔➙➛►▶︎▸⮕\uFFFD�]$/.test(match)) return '->';
                // 左箭头类
                if (/^[←⇐◄◀︎◂⬅]$/.test(match)) return '<-';
                // 双向箭头
                if (/^[↔⇔]$/.test(match)) return '<->';
                // 私用区字符和控制字符 - 移除
                if (/^[\u{E000}-\u{F8FF}]$/u.test(match) || /^[\x00-\x08\x0B\x0C\x0E-\x1F]$/.test(match)) return '';
                // 连续的箭头 - 合并
                if (match.startsWith('->')) return '->';
                if (match.startsWith('<->')) return '<->';
                if (match.startsWith('<-')) return '<-';
                // 连续的点号
                if (match.startsWith('.')) return '...';
                // 连续的短横线
                if (match.startsWith('-')) return '--';
                // 连续的下划线
                if (match.startsWith('_')) return '__';
                // 行尾空白
                if (/^\s+$/.test(match)) return '';
                return match;
            }
        );
    }

    /**
     * 内容转文本（仅稳定字段，用于哈希计算）
     * 排除易变字段：tool_use_id, id, input, cache_control
     * @param {any} content - 消息内容
     * @param {string} role - 消息角色（可选）
     */
    contentToTextForHash(content, role = null) {
        let parts = [];
        if (role) parts.push(role);

        if (!content) return parts.join(' ');
        if (typeof content === 'string') {
            parts.push(this.normalizeTextForHash(content));
            return parts.join(' ');
        }

        if (Array.isArray(content)) {
            for (const c of content) {
                if (typeof c === 'string') {
                    parts.push(this.normalizeTextForHash(c));
                    continue;
                }

                // tool_result 特殊处理：根据策略决定如何计算哈希
                if (c.type === 'tool_result') {
                    const strategy = this.config.toolResultStrategy || 'strict';

                    // ignore 模式：完全跳过 tool_result，不参与哈希计算
                    if (strategy === 'ignore') {
                        continue;
                    }

                    parts.push(c.type);
                    if (c.name) parts.push(c.name);

                    if (strategy === 'strict') {
                        // 严格模式：包含完整内容
                        if (c.content) parts.push(this.contentToTextForHash(c.content));
                    }
                    // name_only 模式：只添加了 type 和 name，不添加 content
                    continue;
                }

                // 只包含稳定字段
                if (c.type) parts.push(c.type);
                // 排除 cache_control：这是动态添加的缓存控制标记，不是消息内容
                if (c.text) parts.push(this.normalizeTextForHash(c.text));
                if (c.thinking) parts.push(this.normalizeTextForHash(c.thinking));
                if (c.name) parts.push(c.name);  // 工具名是稳定的
                // 排除: tool_use_id, id, input（这些每次请求都可能变化）
                if (c.content) parts.push(this.contentToTextForHash(c.content));
            }
            return parts.join(' ');
        }

        // 对于非数组对象，只提取稳定字段
        if (typeof content === 'object') {
            let parts2 = [];
            if (content.type) parts2.push(content.type);
            if (content.text) parts2.push(this.normalizeTextForHash(content.text));
            if (content.name) parts2.push(content.name);
            return parts.concat(parts2).join(' ');
        }

        return parts.join(' ');
    }

    /**
     * Token 计数（使用官方 tokenizer，fallback 到字符估算）
     */
    countTokensSimple(text) {
        if (!text) return 0;
        try {
            return countTokens(text);
        } catch (e) {
            // tokenizer 失败时 fallback，添加警告日志
            if (CACHE_ESTIMATION_CONFIG.DEBUG) {
                console.warn(`[Kiro] Tokenizer failed, using fallback: ${e.message}`);
            }
            return Math.ceil(text.length / 4);
        }
    }

    /**
     * 安全哈希函数（使用 MD5，碰撞概率极低）
     */
    simpleHash(str) {
        return crypto.createHash('md5').update(str, 'utf8').digest('hex');
    }

    /**
     * 获取统计信息
     */
    getStats() {
        return {
            ...this.stats,
            cacheSize: this.requestCache.size,
            config: this.config
        };
    }

    /**
     * 清空缓存
     */
    clearCache() {
        this.requestCache.clear();
        console.log('[Kiro CacheEstimator] Cache cleared');
    }
}

// 按账号+模型+会话隔离的缓存推测器实例 Map<cacheKey, { estimator, lastUsed }>
const accountCacheEstimators = new Map();

// 账号缓存推测器的配置
export const ACCOUNT_CACHE_CONFIG = {
    MAX_ESTIMATORS: 500,         // 最多缓存多少个推测器实例（账号*模型*会话）
    ESTIMATOR_TTL: 600000,       // 推测器的过期时间（10分钟，比缓存TTL长）
    CLEANUP_INTERVAL: 120000,    // 清理间隔（2分钟，更频繁的清理）
    MEMORY_LIMIT_MB: 200,        // 缓存推测器组件的内存使用上限（MB）
    MEMORY_LIMIT_RATIO: 0.3,     // 占总堆内存的最大比例（30%）
    AGGRESSIVE_CLEANUP_THRESHOLD: 0.8  // 达到80%容量时触发激进清理
};

let lastCleanupTime = Date.now();
let lastMemoryCheck = Date.now();

/**
 * 获取当前进程的内存使用情况（MB）
 */
export function getMemoryUsageMB() {
    const usage = process.memoryUsage();
    return usage.heapUsed / 1024 / 1024;
}

/**
 * 清理过期的缓存推测器
 * @param {boolean} aggressive - 是否执行激进清理（忽略TTL，只保留最近使用的）
 */
export function cleanupExpiredAccountEstimators(aggressive = false) {
    const now = Date.now();

    // 检查是否需要清理
    if (!aggressive && now - lastCleanupTime < ACCOUNT_CACHE_CONFIG.CLEANUP_INTERVAL) {
        return;
    }
    lastCleanupTime = now;

    let cleanedCount = 0;
    const entries = Array.from(accountCacheEstimators.entries());

    if (aggressive) {
        // 激进清理：按最后使用时间排序，只保留最近使用的一半
        const keepCount = Math.floor(ACCOUNT_CACHE_CONFIG.MAX_ESTIMATORS / 2);
        entries.sort((a, b) => b[1].lastUsed - a[1].lastUsed);

        for (let i = keepCount; i < entries.length; i++) {
            accountCacheEstimators.delete(entries[i][0]);
            cleanedCount++;
        }

        console.log(`[Kiro CacheEstimator] Aggressive cleanup: removed ${cleanedCount} estimators, kept ${keepCount}`);
    } else {
        // 常规清理：删除过期的
        for (const [cacheKey, entry] of entries) {
            if (now - entry.lastUsed > ACCOUNT_CACHE_CONFIG.ESTIMATOR_TTL) {
                accountCacheEstimators.delete(cacheKey);
                cleanedCount++;
            }
        }

        // 如果超过最大数量，删除最久未使用的
        if (accountCacheEstimators.size > ACCOUNT_CACHE_CONFIG.MAX_ESTIMATORS) {
            const toDeleteCount = accountCacheEstimators.size - ACCOUNT_CACHE_CONFIG.MAX_ESTIMATORS;
            const remainingEntries = Array.from(accountCacheEstimators.entries());
            remainingEntries.sort((a, b) => a[1].lastUsed - b[1].lastUsed);

            for (let i = 0; i < toDeleteCount; i++) {
                accountCacheEstimators.delete(remainingEntries[i][0]);
                cleanedCount++;
            }
        }

        if (cleanedCount > 0) {
            console.log(`[Kiro CacheEstimator] Cleaned up ${cleanedCount} expired estimators, remaining: ${accountCacheEstimators.size}`);
        }
    }

    // 定期检查内存使用
    if (now - lastMemoryCheck > 60000) { // 每分钟检查一次
        lastMemoryCheck = now;
        const memoryUsage = getMemoryUsageMB();

        if (memoryUsage > ACCOUNT_CACHE_CONFIG.MEMORY_LIMIT_MB) {
            console.warn(`[Kiro CacheEstimator] Memory usage (${memoryUsage.toFixed(2)}MB) exceeds limit (${ACCOUNT_CACHE_CONFIG.MEMORY_LIMIT_MB}MB), triggering aggressive cleanup`);
            cleanupExpiredAccountEstimators(true);
        }
    }

    // 检查是否需要触发激进清理
    const utilizationRatio = accountCacheEstimators.size / ACCOUNT_CACHE_CONFIG.MAX_ESTIMATORS;
    if (!aggressive && utilizationRatio > ACCOUNT_CACHE_CONFIG.AGGRESSIVE_CLEANUP_THRESHOLD) {
        console.log(`[Kiro CacheEstimator] Utilization (${(utilizationRatio * 100).toFixed(1)}%) exceeds threshold, triggering aggressive cleanup`);
        cleanupExpiredAccountEstimators(true);
    }
}

/**
 * 生成缓存推测器的唯一键
 * @param {string} accountId - 账号唯一标识（uuid）
 * @param {string} model - 模型名称
 * @param {string} sessionId - 会话ID（可选）
 * @returns {string} 缓存键
 */
export function generateCacheKey(accountId, model, sessionId = null) {
    // 格式: account:model:session 或 account:model（如果没有sessionId）
    const effectiveAccountId = accountId || `default_${process.pid}`;
    const effectiveModel = model || 'default';

    if (sessionId) {
        return `${effectiveAccountId}:${effectiveModel}:${sessionId}`;
    }
    return `${effectiveAccountId}:${effectiveModel}`;
}

/**
 * 获取或创建指定账号+模型+会话的缓存推测器
 * @param {string} accountId - 账号唯一标识（uuid）
 * @param {string} model - 模型名称
 * @param {string} sessionId - 会话ID（可选，用于会话级别隔离）
 * @param {Object} config - 配置选项
 * @returns {KiroCacheEstimator} 该账号+模型+会话的缓存推测器
 */
export function getCacheEstimatorForAccount(accountId, model = null, sessionId = null, config = {}) {
    // 先清理过期的推测器
    cleanupExpiredAccountEstimators();

    // 生成缓存键：account:model:session
    const cacheKey = generateCacheKey(accountId, model, sessionId);

    let entry = accountCacheEstimators.get(cacheKey);

    if (!entry) {
        // 创建新的推测器
        const estimator = new KiroCacheEstimator(config);
        entry = {
            estimator,
            lastUsed: Date.now(),
            accountId: accountId || `default_${process.pid}`,
            model: model || 'default',
            sessionId: sessionId || null
        };
        accountCacheEstimators.set(cacheKey, entry);

        const sessionInfo = sessionId ? ` session: ${sessionId}` : '';
        console.log(`[Kiro] Cache estimator created for account: ${entry.accountId}, model: ${entry.model}${sessionInfo} (total estimators: ${accountCacheEstimators.size})`);
    } else {
        // 更新最后使用时间
        entry.lastUsed = Date.now();
    }

    return entry.estimator;
}

/**
 * 清理指定账号的所有缓存推测器实例
 * 用于提供商不健康或被禁用时同步清理
 * @param {string} accountId - 账号唯一标识（uuid）
 * @returns {number} 清理的实例数量
 */
export function clearCacheEstimatorsForAccount(accountId) {
    if (!accountId) return 0;

    let cleanedCount = 0;
    const keysToDelete = [];

    // 查找所有属于该账号的缓存推测器
    for (const [cacheKey, entry] of accountCacheEstimators.entries()) {
        if (entry.accountId === accountId) {
            keysToDelete.push(cacheKey);
        }
    }

    // 删除找到的所有实例
    for (const key of keysToDelete) {
        accountCacheEstimators.delete(key);
        cleanedCount++;
    }

    if (cleanedCount > 0) {
        console.log(`[Kiro CacheEstimator] Cleared ${cleanedCount} estimator instances for account: ${accountId}`);
    }

    return cleanedCount;
}

/**
 * 清理指定会话的缓存推测器实例
 * 用于粘性会话被删除时同步清理
 * @param {string} sessionId - 会话ID
 * @returns {number} 清理的实例数量
 */
export function clearCacheEstimatorsForSession(sessionId) {
    if (!sessionId) return 0;

    let cleanedCount = 0;
    const keysToDelete = [];

    // 查找所有属于该会话的缓存推测器
    for (const [cacheKey, entry] of accountCacheEstimators.entries()) {
        if (entry.sessionId === sessionId) {
            keysToDelete.push(cacheKey);
        }
    }

    // 删除找到的所有实例
    for (const key of keysToDelete) {
        accountCacheEstimators.delete(key);
        cleanedCount++;
    }

    if (cleanedCount > 0) {
        console.log(`[Kiro CacheEstimator] Cleared ${cleanedCount} estimator instances for session: ${sessionId}`);
    }

    return cleanedCount;
}

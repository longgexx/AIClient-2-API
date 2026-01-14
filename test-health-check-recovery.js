/**
 * 测试健康检查自动恢复功能
 *
 * 测试步骤：
 * 1. 启动服务器（健康检查间隔设置为30秒以便快速测试）
 * 2. 验证定期健康检查是否启动
 * 3. 模拟提供商被标记为不健康
 * 4. 验证健康检查是否自动重试并恢复
 */

import { initializeConfig } from './src/core/config-manager.js';
import { initApiService } from './src/services/service-manager.js';
import { initializeAPIManagement } from './src/services/api-manager.js';
import { getProviderPoolManager } from './src/services/service-manager.js';

async function testHealthCheckRecovery() {
    console.log('=== 健康检查自动恢复测试 ===\n');

    // 1. 初始化配置（使用30秒间隔以便快速测试）
    console.log('[Step 1] 初始化配置...');
    const config = await initializeConfig([]);
    config.HEALTH_CHECK_INTERVAL = 30000; // 30秒间隔用于测试
    console.log(`  健康检查间隔: ${config.HEALTH_CHECK_INTERVAL}ms (30秒)\n`);

    // 2. 初始化服务
    console.log('[Step 2] 初始化服务...');
    const services = await initApiService(config);
    console.log(`  已初始化 ${Object.keys(services).length} 个服务\n`);

    // 3. 启动定期健康检查
    console.log('[Step 3] 启动定期健康检查...');
    const heartbeatAndRefreshToken = initializeAPIManagement(services, config);
    console.log('  定期健康检查已启动\n');

    // 4. 获取 ProviderPoolManager
    const poolManager = getProviderPoolManager();
    if (!poolManager) {
        console.log('  ⚠️  没有找到 ProviderPoolManager，可能没有配置号池\n');
        process.exit(0);
    }

    // 5. 显示当前提供商状态
    console.log('[Step 4] 当前提供商状态:');
    const providerStatus = poolManager.providerStatus;
    let totalProviders = 0;
    let healthyProviders = 0;
    let unhealthyProviders = 0;

    for (const providerType in providerStatus) {
        for (const status of providerStatus[providerType]) {
            totalProviders++;
            if (status.config.isHealthy) {
                healthyProviders++;
                console.log(`  ✓ ${providerType} - ${status.config.uuid}: 健康`);
            } else {
                unhealthyProviders++;
                console.log(`  ✗ ${providerType} - ${status.config.uuid}: 不健康`);
            }
        }
    }
    console.log(`  总计: ${totalProviders} 个提供商 (健康: ${healthyProviders}, 不健康: ${unhealthyProviders})\n`);

    if (totalProviders === 0) {
        console.log('  ⚠️  没有找到任何提供商，测试结束\n');
        process.exit(0);
    }

    // 6. 模拟标记第一个健康的提供商为不健康
    console.log('[Step 5] 模拟提供商故障...');
    let testProvider = null;
    let testProviderType = null;

    for (const providerType in providerStatus) {
        for (const status of providerStatus[providerType]) {
            if (status.config.isHealthy) {
                testProvider = status.config;
                testProviderType = providerType;
                break;
            }
        }
        if (testProvider) break;
    }

    if (!testProvider) {
        console.log('  ⚠️  没有找到健康的提供商用于测试\n');
        process.exit(0);
    }

    console.log(`  选择测试提供商: ${testProviderType} - ${testProvider.uuid}`);

    // 手动标记为不健康
    poolManager.markProviderUnhealthyImmediately(testProviderType, testProvider, '测试：模拟故障');
    console.log(`  ✗ 已标记为不健康\n`);

    // 7. 等待健康检查自动恢复
    console.log('[Step 6] 等待自动恢复...');
    console.log(`  健康检查将在 30 秒后执行`);
    console.log(`  提示：健康检查会跳过健康的提供商，只检查不健康的提供商\n`);

    // 监控恢复状态
    let recovered = false;
    const checkInterval = setInterval(() => {
        const currentStatus = poolManager.providerStatus[testProviderType].find(
            s => s.config.uuid === testProvider.uuid
        );

        if (currentStatus && currentStatus.config.isHealthy) {
            recovered = true;
            console.log(`  ✓ 提供商已自动恢复为健康状态！`);
            console.log(`  恢复时间: ${new Date().toLocaleString()}\n`);

            console.log('=== 测试成功 ===');
            console.log('健康检查自动恢复功能正常工作！\n');

            clearInterval(checkInterval);
            process.exit(0);
        }
    }, 5000); // 每5秒检查一次

    // 60秒超时
    setTimeout(() => {
        if (!recovered) {
            console.log(`  ✗ 超时：60秒内未恢复\n`);
            console.log('=== 测试失败 ===');
            console.log('提供商未能在预期时间内自动恢复\n');
            clearInterval(checkInterval);
            process.exit(1);
        }
    }, 60000);
}

// 运行测试
testHealthCheckRecovery().catch(error => {
    console.error('测试失败:', error);
    process.exit(1);
});

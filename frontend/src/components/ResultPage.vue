<template>
    <div class="result">
        <div v-if="isProcessing" class="loading-overlay">
            <div class="loading-spinner"></div>
            <p>正在处理图片，请稍候...</p>
        </div>
        <div class="half-width">
            <div class="half-height">
                <div class="result-title">
                    <h2 class="title">Dehazing Result (Infrared Image)</h2>
                </div>
                <div class="result-content">
                    <img v-if="dehazingImage" :src="dehazingImage" class="result-image"/>
                </div>
            </div>
            <div class="half-height">
                <div class="result-title">
                    <h2 class="title">Fusion Result (Dehazed Infrared + Origin Thermal)</h2>
                </div>
                <div class="result-content">
                    <img v-if="fusingImage" :src="fusingImage" class="result-image"/>
                </div>
            </div>
        </div>
        <div class="half-width">
            <div class="half-height">
                <div class="result-title">
                    <h2 class="title">Detection Result (Based on Fusion)</h2>
                </div>
                <div class="result-content">
                    <img v-if="combinedImage" :src="combinedImage" class="result-image"/>
                </div>
            </div>
            <div class="half-height">
                <div class="metrics-container">
                    <div class="metrics-section">
                        <h3 class="metrics-title">处理指标统计</h3>
                        
                        <!-- 去雾指标 -->
                        <div v-if="metrics?.dehazing" class="metrics-content">
                            <h4 class="section-title">去雾处理指标</h4>
                            <div class="metric-item">
                                <span class="metric-label">原始梯度:</span>
                                <span class="metric-value">{{ metrics.dehazing.original_gradient.toFixed(4) }}</span>
                            </div>
                            <div class="metric-item">
                                <span class="metric-label">去雾后梯度:</span>
                                <span class="metric-value">{{ metrics.dehazing.dehazed_gradient.toFixed(4) }}</span>
                            </div>
                            <div class="metric-item">
                                <span class="metric-label">改善比例:</span>
                                <span class="metric-value">{{ (metrics.dehazing.improvement_ratio * 100).toFixed(2) }}%</span>
                            </div>
                            <div class="metric-item">
                                <span class="metric-label">处理时间:</span>
                                <span class="metric-value">{{ metrics.dehazing.processing_time.toFixed(2) }}s</span>
                            </div>
                        </div>

                        <!-- 融合指标 -->
                        <div v-if="metrics?.fusion" class="metrics-content">
                            <h4 class="section-title">图像融合指标</h4>
                            <div class="metric-item">
                                <span class="metric-label">熵值:</span>
                                <span class="metric-value">{{ metrics.fusion.entropy.toFixed(2) }}</span>
                            </div>
                            <div class="metric-item">
                                <span class="metric-label">空间频率:</span>
                                <span class="metric-value">{{ metrics.fusion.spatial_frequency.toFixed(2) }}</span>
                            </div>
                            <div class="metric-item">
                                <span class="metric-label">标准差:</span>
                                <span class="metric-value">{{ metrics.fusion.std_deviation.toFixed(2) }}</span>
                            </div>
                            <div class="metric-item">
                                <span class="metric-label">互信息(融合↔热成像):</span>
                                <span class="metric-value">{{ metrics.fusion.mi_thermal.toFixed(2) }}</span>
                            </div>
                            <div class="metric-item">
                                <span class="metric-label">互信息(融合↔红外):</span>
                                <span class="metric-value">{{ metrics.fusion.mi_ir.toFixed(2) }}</span>
                            </div>
                        </div>

                        <!-- 检测指标 -->
                        <div v-if="metrics?.detection" class="metrics-content">
                            <h4 class="section-title">目标检测指标</h4>
                            <div class="metric-item">
                                <span class="metric-label">总检测数量:</span>
                                <span class="metric-value">{{ metrics.detection.total_detections }}</span>
                            </div>
                            <div v-for="(details, clsId) in metrics.detection.detection_details" :key="clsId" class="class-metrics">
                                <h4 class="class-title">{{ details.class_name }}</h4>
                                <div class="metric-item">
                                    <span class="metric-label">检测数量:</span>
                                    <span class="metric-value">{{ details.count }}</span>
                                </div>
                                <div class="metric-item">
                                    <span class="metric-label">平均置信度:</span>
                                    <span class="metric-value">{{ (details.avg_confidence * 100).toFixed(2) }}%</span>
                                </div>
                                <div class="metric-item">
                                    <span class="metric-label">置信度范围:</span>
                                    <span class="metric-value">
                                        {{ (details.confidence_stats.min * 100).toFixed(2) }}% - 
                                        {{ (details.confidence_stats.max * 100).toFixed(2) }}%
                                    </span>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="result-get">
                        <button class="get-btn" @click="downloadAll">Download All</button>
                    </div>
                </div>
            </div>
        </div>
    </div>
</template>
<script setup>
import { ref, onMounted, onUnmounted } from 'vue';
import { ElMessage } from 'element-plus';
import axios from 'axios';
import { useRoute, useRouter } from 'vue-router';

const dehazingImage = ref(null);
const fusingImage = ref(null);
const combinedImage = ref(null);
// const confidenceLevel = ref(90);
const pollingInterval = ref(null);
const isProcessing = ref(true);
const maxPollingAttempts = 20; // 最多轮询20次
let pollingAttempts = 0;
const retryCount = ref(0);
const maxRetries = 3;
const sessionId = ref(null);
const router = useRouter();
const metrics = ref(null);

// 从路由获取会话ID
onMounted(() => {
    const route = useRoute();
    sessionId.value = route.query.session_id;
    if (!sessionId.value) {
        ElMessage.error('会话ID不存在，请返回重新处理图片');
        router.push('/processing');
        return;
    }
    startPolling();
});

// 开始轮询
const startPolling = () => {
    pollingInterval.value = setInterval(async () => {
        try {
            const response = await axios.get(`http://localhost:5000/get-processed-images?session_id=${sessionId.value}`);
            const { 
                dehazingImage: dehazingPath, 
                fusingImage: fusingPath, 
                combinedImage: combinedPath, 
                status,
                metrics: responseMetrics
            } = response.data;
            
            if (status === 'completed') {
                // 所有图片都已处理完成
                dehazingImage.value = `http://localhost:5000${dehazingPath}`;
                fusingImage.value = `http://localhost:5000${fusingPath}`;
                combinedImage.value = `http://localhost:5000${combinedPath}`;
                metrics.value = responseMetrics;
                isProcessing.value = false;
                clearInterval(pollingInterval.value);
                ElMessage.success('图片处理完成！');
                retryCount.value = 0; // 重置重试计数
            } else if (status === 'error') {
                // 处理出错
                clearInterval(pollingInterval.value);
                ElMessage.error('处理出错，请重试');
                isProcessing.value = false;
            } else if (status === 'failed') {
                // 处理失败
                clearInterval(pollingInterval.value);
                ElMessage.error('处理失败，请重试');
                isProcessing.value = false;
            }
            // 如果是processing状态，继续轮询
            
            pollingAttempts++;
            if (pollingAttempts >= maxPollingAttempts) {
                clearInterval(pollingInterval.value);
                ElMessage.error('处理超时，请重试');
                isProcessing.value = false;
            }
        } catch (error) {
            console.error('轮询失败:', error);
            retryCount.value++;
            
            if (retryCount.value >= maxRetries) {
                clearInterval(pollingInterval.value);
                ElMessage.error('服务器连接失败，请确保后端服务正在运行');
                isProcessing.value = false;
            }
        }
    }, 2000); // 每2秒轮询一次
};

// 组件卸载时清除轮询
onUnmounted(() => {
    if (pollingInterval.value) {
        clearInterval(pollingInterval.value);
    }
});

const downloadAll = async () => {
    const images = [
        { src: dehazingImage.value, name: 'dehazing_image.jpg' },
        { src: fusingImage.value, name: 'fusing_image.jpg' },
        { src: combinedImage.value, name: 'detected_image.jpg' },
    ];
    
    for (const image of images) {
        if (image.src) {
            try {
                // 使用fetch获取图片数据
                const response = await fetch(image.src);
                const blob = await response.blob();
                
                // 创建Blob URL
                const blobUrl = window.URL.createObjectURL(blob);
                
                // 创建下载链接
                const link = document.createElement('a');
                link.href = blobUrl;
                link.download = image.name;
                
                // 添加到文档并触发点击
                document.body.appendChild(link);
                link.click();
                
                // 清理
                document.body.removeChild(link);
                window.URL.revokeObjectURL(blobUrl);
                
                // 添加短暂延迟，避免浏览器阻止多个下载
                await new Promise(resolve => setTimeout(resolve, 100));
            } catch (error) {
                console.error(`下载图片失败: ${image.name}`, error);
                ElMessage.error(`下载图片失败: ${image.name}`);
            }
        } else {
            ElMessage.warning(`图片不可用: ${image.name}`);
        }
    }
    
    ElMessage.success('所有图片下载完成');
};
</script>
<style scoped>
.result {
    display: flex;
    flex-direction: row;
    align-items: center;
    justify-content: center;
    margin-top: 2.4%;
    margin-bottom: 2.4%;
    margin-left: 5%;
    height: clamp(0vh, 90%, 80vh);
    width: 90%;
    background-color: #ffffff;
    border-radius: 2.8vh;
}

.half-width {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    width: 50%;
    height: 100%;
}

.half-height {
    display: flex;
    flex-direction: column;
    align-items: center;

    height: 50%;
    width: 100%;
}

.result-content {
    display: flex;
    flex-direction: column;
    height: 80%;
    width: 80%;
    margin-top: -2%;
    background: #ffffff;
    border-radius: 2.8vh;
    border: 3px dashed black;
}

.title {
    display: flex;
    margin-top: 1%;
    height: auto;
    font-size: clamp(0rem, 2.5vw, 2.7vh);
}
.result-text {
    display: flex;
    height:53%;
    width: 100%;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}
.confidence-title {
    display: flex;

    height: auto;
    font-size: clamp(0rem, 3.5vw, 3.7vh);
}
.result-get {
    display: flex;
    height: 47%;
    width: 100%;
    flex-direction: column;
    align-items: center;
}
.get-btn {
    padding: 0.1% 9.9%;
    /* 设置按钮的内边距 */
    font-size: clamp(0rem, 1.6vw, 2.2vh);
    /* 设置字体大小 */
    font-weight: bold;
    /* 设置粗体 */
    border: none;
    /* 移除默认边框 */
    border-radius: 2vh;
    /* 设置圆角 */
    cursor: pointer;
    /* 鼠标悬停时显示为指针 */
    transition: background-color 0.3s;
    height:35%;
    width:50%;
    background-color: #000a33;
    color: #ffffff;
}
.get-btn:hover {
    background-color: #02008b;
}
.result-image {
    max-height: 100%;
    max-width: 100%;
    object-fit: contain;
    height: auto;

}
.loading-overlay {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(255, 255, 255, 0.9);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    z-index: 1000;
}

.loading-spinner {
    width: 50px;
    height: 50px;
    border: 5px solid #f3f3f3;
    border-top: 5px solid #3498db;
    border-radius: 50%;
    animation: spin 1s linear infinite;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.metrics-container {
    display: flex;
    flex-direction: column;
    height: 100%;
    width: 100%;
    padding: 20px;
    box-sizing: border-box;
}

.metrics-section {
    flex: 1;
    overflow-y: auto;
    margin-bottom: 20px;
    padding-right: 10px;
}

.metrics-section::-webkit-scrollbar {
    width: 8px;
}

.metrics-section::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 4px;
}

.metrics-section::-webkit-scrollbar-thumb {
    background: #888;
    border-radius: 4px;
}

.metrics-section::-webkit-scrollbar-thumb:hover {
    background: #555;
}

.metrics-title {
    font-size: clamp(0rem, 2vw, 2.2vh);
    color: #333;
    margin-bottom: 15px;
    text-align: center;
}

.section-title {
    font-size: clamp(0rem, 1.8vw, 2vh);
    color: #000a33;
    margin: 15px 0 10px 0;
    padding-bottom: 5px;
    border-bottom: 2px solid #000a33;
}

.metrics-content {
    background: #f5f5f5;
    border-radius: 10px;
    padding: 15px;
    margin-bottom: 15px;
}

.metrics-content:last-child {
    margin-bottom: 0;
}

.metric-item {
    display: flex;
    justify-content: space-between;
    margin-bottom: 8px;
    font-size: clamp(0rem, 1.4vw, 1.6vh);
    padding: 4px 0;
}

.metric-label {
    color: #666;
    font-weight: 500;
    flex: 1;
}

.metric-value {
    color: #333;
    font-weight: 600;
    text-align: right;
    flex: 1;
}

.class-metrics {
    background: #fff;
    border-radius: 8px;
    padding: 10px;
    margin-top: 10px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.class-title {
    font-size: clamp(0rem, 1.6vw, 1.8vh);
    color: #000a33;
    margin-bottom: 8px;
    padding-bottom: 5px;
    border-bottom: 1px solid #eee;
}

.result-get {
    display: flex;
    justify-content: center;
    padding: 10px 0;
}

.get-btn {
    padding: 10px 20px;
    font-size: clamp(0rem, 1.4vw, 1.6vh);
    font-weight: bold;
    border: none;
    border-radius: 2vh;
    cursor: pointer;
    transition: background-color 0.3s;
    background-color: #000a33;
    color: #ffffff;
    width: 50%;
}

.get-btn:hover {
    background-color: #02008b;
}
</style>
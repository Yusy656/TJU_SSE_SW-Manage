<template>
    <div class="processing">
        <div class="upload">
            <div class="infrared">
                <div class="upload-img" id="left-img" @click="selectImage('infrared')">
                    <img v-if="infraredImage?.base64" :src="infraredImage.base64" class="uploaded-png" />
                    <img v-else src="@/assets/上传.png" class="upload-png" />
                    <p class="upload-text1" v-if="!infraredImage?.base64">Click to upload or drag and drop</p>
                    <p class="upload-text2" v-if="!infraredImage?.base64">SVG, PNG, JPG</p>
                    <input type="file" ref="infraredFileInput" accept="image/*" @change="handleImageUpload('infrared')"
                        style="display: none;" />
                </div>
                <div class="save">
                    <button class="btn" id="save-button" :style="{ marginLeft: '18%' }"
                        @click="saveImage('infrared')">Upload Infrared Image</button>
                </div>
            </div>
            <div class="add">
                <div class="add-img"><img src="@/assets/add.png"></div>
            </div>
            <div class="thermal">
                <div class="upload-img" id="right-img" @click="selectImage('thermal')">
                    <img v-if="thermalImage?.base64" :src="thermalImage.base64" class="uploaded-png" />
                    <img v-else src="@/assets/上传.png" class="upload-png" />
                    <p class="upload-text1" v-if="!thermalImage?.base64">Click to upload or drag and drop</p>
                    <p class="upload-text2" v-if="!thermalImage?.base64">SVG, PNG, JPG</p>
                    <input type="file" ref="thermalFileInput" accept="image/*" @change="handleImageUpload('thermal')"
                        style="display: none;" />
                </div>
                <div class="save">
                    <button class="btn" id="save-button" @click="saveImage('thermal')">Upload Thermal Image</button>
                </div>
            </div>
        </div>
        <div class="get-result">
            <button 
                class="btn" 
                id="get-result" 
                @click="getResults"
                :disabled="!canGetResults"
                :class="{ 'disabled': !canGetResults }"
            >
                {{ canGetResults ? 'Get Results' : 'Please Upload Both Images' }}
            </button>
        </div>
    </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'; // 导入 ref，移除 onMounted
import { ElMessage } from 'element-plus'; // 导入 ElMessage
import { useRouter } from 'vue-router'; // 导入 useRouter
import axios from 'axios'; // 导入 axios

const infraredFileInput = ref(null); // 创建 ref 引用
const thermalFileInput = ref(null); // 创建 ref 引用

const infraredImage = ref(null); // 用于存储红外图像的状态 (对象 { base64, file })
const thermalImage = ref(null); // 用于存储热成像图像的状态 (对象 { base64, file })
const router = useRouter();
const sessionId = ref(null); // 添加会话ID状态

// 在script setup部分添加新的状态变量
const infraredUploaded = ref(false);
const thermalUploaded = ref(false);

// 创建新会话
const createSession = async () => {
    try {
        const response = await axios.post('http://localhost:5000/create-session');
        sessionId.value = response.data.session_id;
        console.log('Session created:', sessionId.value);
    } catch (error) {
        console.error('Failed to create session:', error);
        ElMessage.error({
            message: '创建会话失败，请刷新页面重试',
            customClass: 'mesinfo',
            duration: 1500
        });
    }
};

// 组件挂载时创建会话
onMounted(() => {
    createSession();
});

// 选择图像的方法
const selectImage = (type) => {
    if (type === 'infrared') {
        infraredFileInput.value.click(); // 点击红外区域时触发文件输入
    } else if (type === 'thermal') {
        thermalFileInput.value.click(); // 点击热成像区域时触发文件输入
    }
};

// 处理图像上传的方法
const handleImageUpload = (type) => {
    const fileInput = type === 'infrared' ? infraredFileInput.value : thermalFileInput.value; // 使用 ref 的 value
    const file = fileInput.files[0]; // 获取选择的文件

    if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            if (type === 'infrared') {
                infraredImage.value = { base64: e.target.result, file: file }; // 存储对象
            } else {
                thermalImage.value = { base64: e.target.result, file: file }; // 存储对象
            }
        };
        reader.readAsDataURL(file); // 读取文件内容
    }
};

// 保存图像并上传到后端
const saveImage = async (type) => {
    if (!sessionId.value) {
        ElMessage.error({
            message: '会话未创建，请刷新页面重试',
            customClass: 'mesinfo',
            duration: 1500
        });
        return;
    }

    const imageRef = type === 'infrared' ? infraredImage : thermalImage;
    const file = imageRef.value?.file;

    if (file) {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('type', type);
        formData.append('session_id', sessionId.value);

        try {
            const response = await axios.post('http://localhost:5000/upload', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data'
                }
            });

            console.log('Upload successful:', response.data);
            // 更新对应图片的上传状态
            if (type === 'infrared') {
                infraredUploaded.value = true;
            } else {
                thermalUploaded.value = true;
            }
            
            ElMessage.success({
                message: `${type.charAt(0).toUpperCase() + type.slice(1)} image uploaded successfully!`,
                customClass: 'mesinfo',
                duration: 1500
            });
        } catch (error) {
            console.error('Upload failed:', error);
            ElMessage.error({
                message: `Failed to upload ${type.charAt(0).toUpperCase() + type.slice(1)} image.`,
                customClass: 'mesinfo',
                duration: 1500
            });
        }
    } else {
        console.error(`No ${type} image file selected.`);
        ElMessage.error({
            message: `No ${type.charAt(0).toUpperCase() + type.slice(1)} image selected. Please select an image first!`,
            customClass: 'mesinfo',
            duration: 1500
        });
    }
};

// 添加计算属性来判断是否可以获取结果
const canGetResults = computed(() => {
    return infraredUploaded.value && thermalUploaded.value;
});

const getResults = async () => {
    if (!sessionId.value) {
        ElMessage.error({
            message: '会话未创建，请刷新页面重试',
            customClass: 'mesinfo',
            duration: 1500
        });
        return;
    }

    try {
        const formData = new FormData();
        formData.append('session_id', sessionId.value);
        const response = await axios.post('http://localhost:5000/process', formData);
        console.log('Start process successful:', response.data);
        ElMessage.success({
            message: '开始处理图片',
            customClass: 'mesinfo',
            duration: 1500
        });
        router.push({
            path: '/result',
            query: { session_id: sessionId.value }
        });
    } catch (error) {
        console.error('Process failed:', error);
        ElMessage.error({
            message: "处理失败，请重试",
            customClass: 'mesinfo',
            duration: 1500
        });
    }
}
</script>
<style scoped>
.processing {
    display: flex;
    width: 100%;
    height: 100%;
    flex-direction: column;
    color: #b2b2b2;
}

.upload {
    display: flex;
    flex-direction: row;
    width: 100%;
    height: 65%;
}

.infrared {
    display: flex;
    flex-direction: column;
    width: 46%;
    height: 100%;
}

.add {
    display: flex;
    flex-direction: column;
    width: 8%;
    height: 100%;
}

.add-img {
    display: flex;
    position: relative;
    flex-direction: column;
    justify-content: center;
    margin-top: 0%;
    height: 72.5%;
    align-items: center;
}

.add-img img {
    width: 60%;
    margin-left: -9%;
    margin-top: 130%;
    transform: (-50%, -50%);
}

.thermal {
    display: flex;
    flex-direction: column;
    width: 46%;
    height: 100%;
}

.get-result {
    display: flex;
    flex-direction: column;
    width: 100%;
    height: 35%;
    align-items: start;
    justify-content: start;
}

.btn {
    padding: 0.1% 9.9%;
    /* 设置按钮的内边距 */
    font-size: clamp(0rem, 1.6vw, 2.2vh);
    ;
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

    /* 添加过渡效果 */
}

#get-result:hover {
    background-color: #45a049;
    /* 悬停时背景颜色变化 */
}

#get-result {
    height: 24%;
    margin-top: 5%;
    width: 40%;
    margin-left: 30%;
    background-color: #056717;
    /* 设置按钮背景颜色 */
    color: #ffffff;
    /* 设置内文字颜色 */
}

.upload-img {
    display: flex;
    flex-direction: column;
    align-items: center;
    background: #ffffff;
    border-radius: 2.8vh;
    width: 82%;
    height: 60%;
    margin-top: 16%;
    border: 3px dashed black;
    /* 设置虚线边框，4px 为边框宽度，黑色 */
    color: #b2b2b2;
    cursor: pointer;
}

#left-img {
    display: flex;
    margin-right: 0%;
    margin-left: 18%;
}

.upload-png {
    display: flex;
    margin-top: 10%;
    max-height: 30%;
    height: auto;
    width: auto;
    object-fit: contain;
    /* 保持原图比例 */
}

.uploaded-png {
    display: flex;
    margin-top: 0%;
    max-height: 100%;
    max-width: 100%;
    height: auto;
    width: auto;
    object-fit: contain;
    /* 保持原图比例 */
}

.upload-text1 {
    display: flex;
    height: 25%;
    margin-top: 5%;
    font-size: clamp(0rem, 1.6vw, 2.2vh);
    ;
    font-weight: bold;
    /* 设置粗体 */
}

.upload-text2 {
    display: flex;
    margin-top: 1%;
    margin-bottom: 0%;
    height: 29%;
    font-size: clamp(0rem, 1.6vw, 2.2vh);
    ;
    /* 设置字体大小 */
    font-weight: normal;
    /* 设置粗体 */
}

.save {
    display: flex;
    width: 100%;
    height: 24%;
    align-items: center;
}

#save-button {
    margin-top: 5%;
    background-color: #000a33;
    /* 设置按钮背景颜色 */
    color: #ffffff;
    /* 设置内文字颜色 */
    height: 78%;
    width: 82%;
}

#save-button:hover {
    background-color: #02008b;
    /* 悬停时背景颜色变化 */
}

.infrared-img {
    height: 100%;

}

.thermal-img {
    height: 100%;
}

#get-result.disabled {
    background-color: #cccccc;
    cursor: not-allowed;
    opacity: 0.7;
}

#get-result.disabled:hover {
    background-color: #cccccc;
}
</style>
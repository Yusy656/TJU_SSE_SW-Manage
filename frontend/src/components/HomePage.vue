<template>
  <div class="home">
    <h1 class="title">Smoke Environment Human Target Detection Development</h1>
    <div class="description">
      <p>Remove fog from images and label human figures using deep learning. </p>
      <p>Upload your image or video and see the magic.</p>
    </div>
    <div class="buttons">
      <button class="start-button" @click="goToProcessing">Get Started</button>
      <button class="learn-more-button" @click="goToLearnMore">Learn More</button>
    </div>
  </div>

</template>

<script setup>
import { useRouter } from 'vue-router';

const router = useRouter();
const goToLearnMore = () => {
  // 跳转到 LearnMore 页面
  router.push('/learn-more'); 
}
const goToProcessing = () => {
  // 跳转到 Processing 页面
  router.push('/processing'); 
}
</script>

<style scoped>
.home {
  display: flex;
  width: 100%;
  height: 100%;
  flex-direction: column;
  color: #ffffff;
  /* 设置字体颜色为白色 */
  /* 您可以根据需要添加更多样式 */
}

.title {
  margin-left: 15%;
  margin-top: 13%;
  width: 70%;
  height: auto;
  text-align: center;
  font-size: 3vw;
  /* 使用视口宽度单位，使字号随窗口大小变化 */
}

.description {
  margin-left: 17.5%;
  margin-top: 0%;
  width: 65%;
  height: auto;
  text-align: center;
  font-size: 1.6vw;
}
.buttons {
  display: flex; /* 使用 Flexbox 布局排列按钮 */
  justify-content: center; /* 水平居中对齐按钮 */
  gap: 2%; /* 按钮之间的间距 */
}

button {
  padding: 1.9% 5.9%; /* 设置按钮的内边距 */
  font-size: 1.6vw; /* 设置字体大小 */
  font-weight: bold; /* 设置粗体 */
  border: none; /* 移除默认边框 */
  border-radius: 2vh; /* 设置圆角 */
  
  cursor: pointer; /* 鼠标悬停时显示为指针 */
  transition: background-color 0.3s; /* 添加过渡效果 */
}

/* 悬停效果 */
.start-button:hover {
  background-color: #45a049; /* 悬停时背景颜色变化 */
}
.learn-more-button:hover {
  background-color: #c3c3c3; /* 悬停时背景颜色变化 */
}
.start-button {
  background-color: #056717; /* 设置按钮背景颜色 */
  color: #ffffff; /* 设置内文字颜色 */
}
.learn-more-button {
  background-color: #ffffff; /* 设置按钮背景颜色 */
  color: #000000; /* 设置内文字颜色 */
}
</style>
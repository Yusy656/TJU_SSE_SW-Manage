<template>
  <div id="app">
    <div class="title">
      <img src="./assets/eye.png" alt="Eye" class="eye" />
      <span class="title-text">EmberHunter</span>
      <div class="right-buttons">
        <button class="gotoBoard" @click="goToPage">{{ buttonText }}&#8594;</button> <!-- 按钮文本根据路由动态变化 -->
      </div>
    </div>
    <div class="content">
      <nav>
        <router-link to="/"></router-link>
        <router-link to="/learn-more"></router-link>
      </nav>
      <router-view></router-view> <!-- 用于显示当前路由组件 -->
    </div>
  </div>

</template>

<script setup>
import { computed } from 'vue';
import { useRoute, useRouter } from 'vue-router';

// 获取当前路由
const route = useRoute();
const router = useRouter();

// 计算属性，根据当前路由动态返回按钮文本
const buttonText = computed(() => {
  if (route.path === '/') {
    return 'DashBoard';
  } else if (route.path === '/result') {
    return 'Process';
  } else {
    return 'Home';
  }
});

// 点击按钮时跳转到主页
const goToPage = () => {
  if (route.path === '/learn-more'||route.path === '/processing') {
    router.push('/'); // 如果当前是 learn-more，点击后跳转到首页 /
  } else if(route.path === '/result'){
    router.push('/processing'); 
  } else {
    // 其他逻辑可以在这里处理，如返回到 Dashboard 或其他
    router.push('/result');
  }
};
</script>

<style scoped>
#app {
  display: flex;
  flex-direction: column;
  /* 使子元素垂直排列 */
  /* 使用弹性盒子布局 */
  background-image: url('./assets/fireBackground.jpg');
  background-size: cover;
  background-position: center;
  background-repeat: no-repeat;
  height: 100vh;
  position: relative;
  font-family: Arial Narrow Bold, sans-serif, sans-serif;
}

.title {
  display: flex;
  align-items: center;
  margin-left: 0%;
  margin-top: 1.8%;
  width: 100%;
  height: 6%;
}

.eye {
  margin-left: 2%;
  /* 可选，给图标添加一些边距以避免与边缘太近 */
  margin-top: 0%;
  /* 可选，给图标添加一些边距以避免与边缘太近 */
  max-height: unset;
  height: 130%;
  width: auto;
}

.right-buttons {
  display: flex;
  justify-content: flex-end;
  margin-left: auto;
  height: 100%;
}

.gotoBoard {
  margin-right: 2vh;
  width: 25vh;
  font-size: clamp(0rem, 1.4vw, 2.7vh);
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
  background-color: #ffffff;
  /* 设置按钮背景色 */
  color: #c38b00;
  /* 设置字体颜色 */
  text-align: center;
  /* 居中 */
}

.gotoBoard:hover {
  background-color: #c3c3c3;
  /* 悬停时背景颜色变化 */
}

.title-text {
  margin-left: 1%;
  /* 图标和文字之间的空隙 */
  font-size: clamp(0rem, 3vw, 4.5vh);
  /* 使用视口宽度单位，使字号随窗口大小变化 */
  font-weight: bold;
  /* 设置字体为加粗 */
  color: #bababa;
  /* 设置字体颜色为白色 */
  /* 您可以根据需要更改颜色 */
}

.content {
  width: 100%;
  height: 92.2%
}
</style>

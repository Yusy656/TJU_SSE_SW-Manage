// router.js
import { createRouter, createWebHistory } from 'vue-router';
import HomePage from '../components/HomePage.vue'; // 主页组件
import LearnMorePage from '../components/LearnMorePage.vue'; // 新组件
import ProcessingPage from '../components/ProcessingPage.vue'; // 处理页面组件
import ResultPage from '../components/ResultPage.vue'; // 结果页面组件
const routes = [
  { path: '/', component: HomePage }, // 设置主页路由
  { path: '/learn-more', component: LearnMorePage }, // 设置学习更多页面路由
  { path: '/processing', component: ProcessingPage },
  { path: '/result', component: ResultPage },
];

const router = createRouter({
  history: createWebHistory(), // 使用 HTML5 模式
  routes,
});

export default router;
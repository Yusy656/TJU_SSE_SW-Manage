import { createApp } from 'vue'
import App from './App.vue'
import router from './router/router'; 
import 'element-plus/theme-chalk/el-message.css';
import './assets/styles/msg.scss'

const app = createApp(App);
app.use(router);
app.mount('#app');

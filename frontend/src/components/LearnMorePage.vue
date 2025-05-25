<template>
    <div class="learn-more-page">
        <div class="uptext">
            <div class="left-text">
                <h2 class="title">What is EmberHunter？</h2>
                <div class="text">
                    EmberHunter aims to provide an efficient
                    and accurate solution for human target
                    detection in smoky environments for fire
                    rescue robots through advanced computer
                    vision and deep learning technologies.
                </div>
            </div>
            <div class="right-text">
                <h2 class="title">Why EmberHunter？</h2>
                <div class="text">
                    EmberHunter propose an innovative human
                    target detection system that combines dehazing
                    algorithms with multimodal fusion,enabling
                    reliable human detection in visually limited smoky environments and giving fire rescue robots the
                    ability to "see through smoke."
                </div>
            </div>
        </div>
        <div class="downtext">
            <div class="left-text">
                <h2 class="title">How it works？</h2>
                <div class="text">
                    The dehazing algorithm PTTD, combined with physics and deep learning, significantly enhancing image
                    quality. The human detection algorithm YOLOv11, after optimization, adapts well to occluded scenes.
                    The dual-modal fusion approach, utilizing adaptive weights and a Transformer model.
                </div>
            </div>
            <div class="right-text">
                <img src="@/assets/example.png" class="example-png">
            </div>
        </div>
        <div class="group-name" @click="openLink">
            <div class="group-name-png"><img src="@/assets/github.png"></div>
            <div class="group-name-text">No Other Names</div>
        </div>
    </div>
</template>
<script setup>
const openLink = () => {
    window.open('https://github.com/Yusy656/TJU_SSE_SW-Manage', '_blank');
}
</script>

<style scoped>
.learn-more-page {
    display: flex;
    margin-top: 1.8%;
    width: 100%;
    height: 100%;
    flex-direction: column;
    background-color: rgb(255, 255, 255);
    color: black;
}

.uptext {
    display: flex;
    height: 46.5%;
    flex-direction: row;
}

.downtext {
    display: flex;
    height: 46.5%;
    background-color: rgb(218, 218, 218);
    flex-direction: row;
}

.group-name {
    display: flex;
    height: 7%;
    justify-content: flex-end;
    font-weight: bold;
    font-size: 3vh;
    align-items: center; /* 垂直居中对齐 */
    cursor: pointer;  /* 鼠标移入时变色 */
    transition: background-color 0.3s ease; /* 添加过渡效果 */
}

.group-name:hover {
    color:rgb(156, 107, 11)
}

.group-name-text {
    margin-right: 1%;
    /* 文本和图片之间的间距，可根据需要调整 */
}

.group-name-png {
    display: flex;
    /* 保证里面的内容可以使用 Flexbox */
    justify-content: flex-end;
    margin-right: 0.9%;
    height: 100%;
    width: auto;
    align-items: center;
    /* 图片和文本之间的间距，可根据需要调整 */
}

.group-name-png img {
    display: flex;
    max-height: 60%;
    /* 最大高度为父容器的 100% */
    width: auto;
    /* 宽度自适应，保持原比例 */
    height: auto;
    /* 自动设置高度以保持纵横比 */
}

.left-text {
    display: flex;
    flex-direction: column;
    width: 50%;
    height: 100%;
    justify-content: center;
}

.right-text {
    display: flex;
    flex-direction: column;
    width: 50%;
    height: 100%;
    justify-content: center;
}

.title {
    display: flex;
    margin-top: 0%;
    justify-content: center;
    font-size: clamp(0rem, 3.5vw, 3.7vh);
    height: 4%;
    font-weight: bold;
}

.text {
    display: flex;
    margin-top: 2%;
    justify-content: center;
    font-size: clamp(0rem, 2.5vw, 2.7vh);
    width: 75%;
    margin-left: 12.5%;
}
.example-png {
    display: flex;
    max-height: 80%;
    object-fit: contain;
}
</style>
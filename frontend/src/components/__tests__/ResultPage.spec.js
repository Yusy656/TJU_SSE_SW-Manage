/** @jest-environment jsdom */
// 必须先 Mock 静态资源路径
jest.mock('@/assets/images/fireBackground.jpg', () => 'mocked-fire-background-url');

import { mount, flushPromises } from '@vue/test-utils';
import ResultPage from '../ResultPage.vue';
import { ElMessage } from 'element-plus';

// Mock localStorage 实现
let mockLocalStorage = {};
jest.spyOn(Storage.prototype, 'getItem').mockImplementation(key => mockLocalStorage[key]);
jest.spyOn(Storage.prototype, 'setItem').mockImplementation((key, value) => {
  mockLocalStorage[key] = value;
});

// Mock ElMessage.warning
jest.mock('element-plus', () => ({
  ElMessage: {
    warning: jest.fn(),
  }
}));

describe('ResultPage.vue', () => {
  const realCreateElement = document.createElement.bind(document);

  beforeEach(() => {
    // 清理 body
    document.body.innerHTML = '';
    document.body.querySelectorAll('a').forEach(el => el.remove());

    // 初始化 mockLocalStorage 和 ElMessage
    mockLocalStorage = {};
    ElMessage.warning.mockClear();

    // mock createElement，仅针对 <a> 返回 mock click 的元素
    jest.spyOn(document, 'createElement').mockImplementation((tag) => {
      if (tag === 'a') {
        const el = realCreateElement('a');
        el.click = jest.fn();
        return el;
      }
      return realCreateElement(tag);
    });
  });

  it('renders images correctly when localStorage has image data', async () => {
    const dummyPath = 'mocked-fire-background-url';

    mockLocalStorage['dehazingImage'] = dummyPath;
    mockLocalStorage['fusingImage'] = dummyPath;
    mockLocalStorage['combinedImage'] = dummyPath;

    // 避免 onMounted 设置默认值
    jest.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {});

    const wrapper = mount(ResultPage);
    await flushPromises();

    const imgs = wrapper.findAll('img.result-image');
    expect(imgs).toHaveLength(3);
    imgs.forEach(img => {
      expect(img.attributes('src')).toBe(dummyPath);
    });

    expect(wrapper.text()).toContain('Confidence Level: 90%');
    expect(ElMessage.warning).not.toHaveBeenCalled();
  });

  it('shows warning when no images in localStorage', async () => {
    // 清空 localStorage 并禁用 setItem 避免污染
    mockLocalStorage = {};
    jest.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {});

    const wrapper = mount(ResultPage);
    await flushPromises();

    // 确保 dehazingImage、fusingImage、combinedImage 都为空
    expect(wrapper.vm.dehazingImage).toBeFalsy();  // 使用 toBeFalsy 而不是 toBeNull
    expect(wrapper.vm.fusingImage).toBeFalsy();
    expect(wrapper.vm.combinedImage).toBeFalsy();

    expect(ElMessage.warning).toHaveBeenCalledWith('Please upload images first.');
  });

  it('downloadAll triggers downloads for available images', async () => {
    const dummyPath = 'mocked-fire-background-url';

    mockLocalStorage['dehazingImage'] = dummyPath;
    mockLocalStorage['fusingImage'] = dummyPath;
    mockLocalStorage['combinedImage'] = dummyPath;

    jest.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {});

    const wrapper = mount(ResultPage);
    await flushPromises();

    const appendChildMock = jest.spyOn(document.body, 'appendChild');
    const removeChildMock = jest.spyOn(document.body, 'removeChild');

    await wrapper.find('button.get-btn').trigger('click');

    expect(appendChildMock).toHaveBeenCalledTimes(3);
    expect(removeChildMock).toHaveBeenCalledTimes(3);

    const links = [...document.body.querySelectorAll('a')];
    links.forEach(link => {
      expect(link.click).toHaveBeenCalled();
    });

    appendChildMock.mockRestore();
    removeChildMock.mockRestore();
  });

  it('downloadAll warns if image src missing', async () => {
    // 只模拟一个 null 值
    mockLocalStorage['dehazingImage'] = 'img1.jpg';
    mockLocalStorage['fusingImage'] = null;
    mockLocalStorage['combinedImage'] = 'img3.jpg';

    jest.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {});

    const wrapper = mount(ResultPage);
    await flushPromises();

    const appendChildMock = jest.spyOn(document.body, 'appendChild');
    const removeChildMock = jest.spyOn(document.body, 'removeChild');

    await wrapper.find('button.get-btn').trigger('click');

    expect(appendChildMock).toHaveBeenCalledTimes(2); // fusingImage 是 null，不会下载
    expect(removeChildMock).toHaveBeenCalledTimes(2);

    const links = [...document.body.querySelectorAll('a')];
    links.forEach(link => {
      expect(link.click).toHaveBeenCalled();
    });

    expect(ElMessage.warning).toHaveBeenCalledWith('Image not available for download: fusing_image.jpg');

    appendChildMock.mockRestore();
    removeChildMock.mockRestore();
  });
});
import { shallowMount } from '@vue/test-utils'
import App from '../App.vue'

describe('App.vue', () => {
  it('renders properly', () => {
    const wrapper = shallowMount(App)
    expect(wrapper.exists()).toBe(true)
  })

  it('has router view', () => {
    const wrapper = shallowMount(App)
    expect(wrapper.find('router-view').exists()).toBe(true)
  })
}) 
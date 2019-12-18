import axios from 'axios';

axios.defaults.baseURL = "http://127.0.0.1:5000/"

export const AnalyticsAPI = {
  getDemoList: () => axios.get('/demo_list'),
}
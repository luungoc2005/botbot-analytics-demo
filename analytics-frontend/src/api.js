import axios from 'axios';
import { stringify } from 'querystring';

axios.defaults.baseURL = "http://127.0.0.1:5000/"

export const AnalyticsAPI = {
  getDemoList: () => axios.get('/demo_list'),
  getClusteringVisualize: (params = {
      file: '',
      only_fallback: false,
    }) => axios.get(`/clustering_visualize?${stringify(params)}`)
}
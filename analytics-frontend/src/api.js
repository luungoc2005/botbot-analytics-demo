import axios from 'axios';
import { stringify } from 'querystring';

axios.defaults.baseURL = "http://127.0.0.1:5000/"

export const AnalyticsAPI = {
  getDemoList: () => axios.get('/demo_list'),
  getClusteringVisualize: (params = {
      file: '',
      only_fallback: false,
    }) => axios.get(`/clustering_visualize?${stringify(params)}`),
  getIntentsList: (params = {
      file: '',
    }) => axios.get(`/intents_list?${stringify(params)}`),
  getTopIntents: (params = {
      file: '',
      only: '',
      top_n: 10,
    }) => axios.get(`/top_intents?${stringify(params)}`),
  getTopWords: (params = {
    file: '',
    only: '',
    top_n: 10,
  }) => axios.get(`/top_words?${stringify(params)}`),
  getWordsTrend: (params = {
    file: '',
    period: 'D',
    words: '',
  }) => axios.get(`/words_trend?${stringify(params)}`),
  getIntentsTrend: (params = {
    file: '',
    period: 'D',
    intents: '',
  }) => axios.get(`/intents_trend?${stringify(params)}`),
}
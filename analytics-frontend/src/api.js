import axios from 'axios';
import { stringify } from 'querystring';
import io from 'socket.io-client'

const BASE_URL = "http://127.0.0.1:5000/"

export const socket = io(BASE_URL)
axios.defaults.baseURL = BASE_URL

let tasks = []

socket.on('connect', () => {
  socket.emit('connected', JSON.stringify(tasks));
  console.log('Client connected', socket)
})

// socket.on('message', (...args) => console.log(...args))

export const awaitTaskResult = (task_id, callback) => {
  tasks.push(task_id)
  const wrappedCallback = (data) => {
    const json_data = JSON.parse(data)
    if (json_data.task_id === task_id) {
      if (tasks.indexOf(json_data.task_id) > -1) 
        callback(json_data.data)
      
      tasks = tasks.filter(item => item !== task_id)
      socket.off('message', wrappedCallback);
    }
  }
  socket.on('message', wrappedCallback);

  return () => {
    console.log(`Canceling task_id: ${task_id}`)
    tasks = tasks.filter(item => item !== task_id)
    socket.off('message', wrappedCallback); 
  }
}

export const AnalyticsAPI = {
  getDemoList: () => axios.get('/demo_list'),
  getClusteringVisualize: (params = {
      file: '',
      only_fallback: false,
      sid: '',
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
    sid: '',
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
  getSimilarWords: (params = {
    file: '',
    word: '',
    top_n: 10,
  }) => axios.get(`/top_similar_words?${stringify(params)}`),

  getDemoTrainingList: () => axios.get('/demo_training_list'),
  getTrainingStats: (params = {
    file: '',
    sid: '',
  }) => axios.get(`/training_stats?${stringify(params)}`)
}
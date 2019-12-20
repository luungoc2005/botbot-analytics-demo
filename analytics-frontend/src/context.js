import React, { createContext } from 'react';


export const AppContext = createContext({
  demoList: null,
  demoFile: '',
  setDemoFile: () => null,
  demoTrainingList: null,
  demoTrainingFile: '',
  setDemoTrainingFile: () => null,
})

export default AppContext;
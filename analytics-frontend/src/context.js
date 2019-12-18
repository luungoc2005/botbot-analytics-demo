import React, { createContext } from 'react';


export const AppContext = createContext({
  demoList: null,
  demoFile: '',
  setDemoFile: () => null,
})

export default AppContext;
import React, { useState, useEffect } from 'react';
import {
  BrowserRouter as Router,
  Switch,
  Route,
  Link
} from "react-router-dom";

import { AnalyticsAPI } from './api';
import { AppContext } from './context';

import { Navigation } from './components/Navigation';
import { HomePage } from './pages/HomePage';
import { ClusteringPage } from './pages/ClusteringPage';

import 'office-ui-fabric-core/dist/css/fabric.min.css';
import { initializeIcons } from 'office-ui-fabric-react/lib/Icons';

initializeIcons();

function App() {
  const [demoList, setDemoList] = useState(null);
  const [demoFile, setDemoFile] = useState('');

  useEffect(() => {
    const fetchDemoList = async () => {
      const resp = await AnalyticsAPI.getDemoList();
      setDemoList(resp.data);
    }
    fetchDemoList();
  }, []);

  return (
    <Router>
      <AppContext.Provider value={{
        demoList,
        demoFile,
        setDemoFile,
      }}>
        <div className="ms-Grid" dir="ltr">
          <div className="ms-Grid-row">
            <div className="ms-Grid-col ms-sm3"><Navigation /></div>
            <div className="ms-Grid-col ms-sm9">
              <Switch>
                <Route exact path="/" component={HomePage} />
                <Route exact path="/clustering" component={ClusteringPage} />
              </Switch>
            </div>
          </div>
        </div>
      </AppContext.Provider>
    </Router>
  );
}

export default App;

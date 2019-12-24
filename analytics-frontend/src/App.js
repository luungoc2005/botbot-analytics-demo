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
import { InsightsPage } from './pages/InsightsPage';
import { TrendsKeywordsPage } from './pages/TrendsKeywordsPage';
import { TrendsIntentsPage } from './pages/TrendsIntentsPage';
import { TrainingStatsPage } from './pages/TrainingStatsPage';

import 'office-ui-fabric-core/dist/css/fabric.min.css';
import { initializeIcons } from 'office-ui-fabric-react/lib/Icons';

initializeIcons();

function App() {
  const [demoList, setDemoList] = useState(null);
  const [demoTrainingList, setDemoTrainingList] = useState(null);
  const [demoFile, setDemoFile] = useState('');
  const [demoTrainingFile, setDemoTrainingFile] = useState('');

  useEffect(() => {
    const fetchDemoList = async () => {
      const resp = await AnalyticsAPI.getDemoList();
      setDemoList(resp.data);
    }
    fetchDemoList();
  }, []);

  useEffect(() => {
    const fetchDemoTrainingList = async () => {
      const resp = await AnalyticsAPI.getDemoTrainingList();
      setDemoTrainingList(resp.data);
    }
    fetchDemoTrainingList();
  }, []);

  return (
    <Router>
      <AppContext.Provider value={{
        demoList,
        demoFile,
        setDemoFile,
        demoTrainingList,
        demoTrainingFile,
        setDemoTrainingFile,
      }}>
        <div className="ms-Grid" dir="ltr">
          <div className="ms-Grid-row">
            <div className="ms-Grid-col ms-sm3">
              <Navigation />
            </div>
            <div className="ms-Grid-col ms-sm9">
              <Switch>
                <Route exact path="/" component={HomePage} />
                <Route exact path="/clustering" component={ClusteringPage} />
                <Route exact path="/insights" component={InsightsPage} />
                <Route exact path="/trends_keywords" component={TrendsKeywordsPage} />
                <Route exact path="/trends_intents" component={TrendsIntentsPage} />
                <Route exact path="/training" component={TrainingStatsPage} />
              </Switch>
            </div>
          </div>
        </div>
      </AppContext.Provider>
    </Router>
  );
}

export default App;

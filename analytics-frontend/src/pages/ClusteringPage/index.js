import React, { useContext, useEffect, useState } from 'react';

import { DemoFileSelector } from '../../components/DemoFileSelector';

import Plot from 'react-plotly.js'

import { AppContext } from '../../context';
import { AnalyticsAPI } from '../../api';

export const ClusteringPage = () => {
  const { demoFile, setDemoFile } = useContext(AppContext);
  const [ clusteringData, setClusteringData ] = useState(null);

  useEffect(() => {
    if (!demoFile) {
      setClusteringData(null);
      return;
    }

    const fetchClusteringData = async () => {
      const resp = await AnalyticsAPI.getClusteringVisualize({ 
        file: demoFile, 
      })
      setClusteringData(resp.data);
    }
    fetchClusteringData()
  }, [demoFile])

  return <>
    <DemoFileSelector onDemoFileClick={setDemoFile} />
    {clusteringData && <Plot
      data={clusteringData.data}
      layout={clusteringData.layout}
    />}
  </>
}

export default ClusteringPage;
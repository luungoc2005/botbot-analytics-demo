import React, { useContext, useEffect, useState } from 'react';

import { ChoiceGroup } from 'office-ui-fabric-react/lib/ChoiceGroup';
// import { ProgressIndicator } from 'office-ui-fabric-react/lib/ProgressIndicator';
// import { Text } from 'office-ui-fabric-react/lib/Text';
import { Spinner } from 'office-ui-fabric-react/lib/Spinner';
import { Stack } from 'office-ui-fabric-react/lib/Stack';
// import { Label } from 'office-ui-fabric-react/lib/Label';

import { DemoFileSelector } from '../../components/DemoFileSelector';

import Plot from 'react-plotly.js';

import { AppContext } from '../../context';
import { AnalyticsAPI } from '../../api';

export const ClusteringPage = () => {
  const { demoFile, setDemoFile } = useContext(AppContext);
  const [ isOnlyFallback, setIsOnlyFallback ] = useState(true);
  const [ clusteringData, setClusteringData ] = useState(null);
  const [ isClusteringLoading, setIsClusteringLoading ] = useState(false);

  useEffect(() => {
    if (!demoFile) {
      setClusteringData(null);
      return;
    }

    const fetchClusteringData = async () => {
      setIsClusteringLoading(true);
      const resp = await AnalyticsAPI.getClusteringVisualize({ 
        file: demoFile, 
        only_fallback: isOnlyFallback,
      })
      setClusteringData(resp.data);
      setIsClusteringLoading(false);
    }
    fetchClusteringData()
  }, [demoFile, isOnlyFallback])

  return <Stack tokens={{
    childrenGap: 20,
  }}>
    <div>
      <DemoFileSelector onDemoFileClick={setDemoFile} />
    </div>

    {demoFile && <div><ChoiceGroup 
      defaultSelectedKey={isOnlyFallback ? 'A' : 'B'}
      options={[
        {
          key: 'A',
          text: 'Only Fallback',
          iconProps: { iconName: 'Filter' },
        },
        {
          key: 'B',
          text: 'All Messages',
          iconProps: { iconName: 'ClearFilter' },
        }
      ]}
      onChange={(e, option) => setIsOnlyFallback(option.key === 'A')}
    /></div>}

    {/* {isClusteringLoading && <ProgressIndicator 
      label="Loading" 
      description="Visualizing your data" 
    />} */}

    {isClusteringLoading && 
      <Spinner 
        label="Visualizing your data..." 
        ariaLive="assertive" 
        labelPosition="left"
        style={{
          justifyContent: 'left'
        }}
      />}

    {clusteringData && <div><Plot
      data={clusteringData.data}
      layout={clusteringData.layout}
      config={{
        responsive: true
      }}
    /></div>}
  </Stack>
}

export default ClusteringPage;
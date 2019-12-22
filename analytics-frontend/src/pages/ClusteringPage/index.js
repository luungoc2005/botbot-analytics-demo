import React, { useContext, useEffect, useState, useRef } from 'react';

import { ChoiceGroup } from 'office-ui-fabric-react/lib/ChoiceGroup';
// import { ProgressIndicator } from 'office-ui-fabric-react/lib/ProgressIndicator';
// import { Text } from 'office-ui-fabric-react/lib/Text';
import { Spinner } from 'office-ui-fabric-react/lib/Spinner';
import { Stack } from 'office-ui-fabric-react/lib/Stack';
// import { Label } from 'office-ui-fabric-react/lib/Label';

import { DemoFileSelector } from '../../components/DemoFileSelector';

import Plot from 'react-plotly.js';

import { AppContext } from '../../context';
import { AnalyticsAPI, awaitTaskResult, socket } from '../../api';

export const ClusteringPage = () => {
  const { demoFile, setDemoFile } = useContext(AppContext);
  const [ isOnlyFallback, setIsOnlyFallback ] = useState(true);
  const [ clusteringData, setClusteringData ] = useState(null);
  const [ isClusteringLoading, setIsClusteringLoading ] = useState(false);
  const _chartContainerElement = useRef();

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
        sid: socket.id,
      })
      const task_id = resp.data.task_id

      if (task_id) {
        awaitTaskResult(task_id, (data) => {
          // console.log(data)
          setClusteringData(data);
          setIsClusteringLoading(false);
        });
      }
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

    <div ref={_chartContainerElement}>
      {isClusteringLoading && 
        <Spinner 
          label="Visualizing your data. This might take a while..." 
          ariaLive="assertive" 
          labelPosition="left"
          style={{
            justifyContent: 'left'
          }}
        />}

      {clusteringData && <Plot
        data={clusteringData.data}
        layout={{
          width: _chartContainerElement.current ? _chartContainerElement.current.clientWidth : 400,
          height: _chartContainerElement.current ? _chartContainerElement.current.clientWidth * .75 : 300,
          ...clusteringData.layout
        }}
        config={{
          responsive: true
        }}
      />}
    </div>
  </Stack>
}

export default ClusteringPage;
import React, { useContext, useEffect, useState, useRef, useMemo } from 'react';

// import { ChoiceGroup } from 'office-ui-fabric-react/lib/ChoiceGroup';
// import { ProgressIndicator } from 'office-ui-fabric-react/lib/ProgressIndicator';
import { Text } from 'office-ui-fabric-react/lib/Text';
import { Spinner } from 'office-ui-fabric-react/lib/Spinner';
import { Stack } from 'office-ui-fabric-react/lib/Stack';
// import { Label } from 'office-ui-fabric-react/lib/Label';
// import { DetailsList, Selection } from 'office-ui-fabric-react/lib/DetailsList';
import { List } from 'office-ui-fabric-react/lib/List';
// import { Checkbox } from 'office-ui-fabric-react/lib/Checkbox';
// import { Callout } from 'office-ui-fabric-react/lib/Callout';
// import { DefaultButton } from 'office-ui-fabric-react/lib/Button';
// import { TextField } from 'office-ui-fabric-react/lib/TextField';
// import { ComboBox } from 'office-ui-fabric-react/lib/ComboBox';
import { FocusZone } from 'office-ui-fabric-react/lib/FocusZone';
import { Pivot, PivotItem } from 'office-ui-fabric-react/lib/Pivot';
import { MessageBar, MessageBarType } from 'office-ui-fabric-react/lib/MessageBar';

import { mergeStyleSets, getTheme, normalize, getFocusStyle } from 'office-ui-fabric-react/lib/Styling';

import { DemoTrainingFileSelector } from '../../components/DemoTrainingFileSelector';

import Plot from 'react-plotly.js'

import { AppContext } from '../../context';
import { AnalyticsAPI, awaitTaskResult, socket } from '../../api';

const problemsData = {
  'empty_intents': {
    title: 'Empty Intents',
    description: 'Some intents has no training examples. These intents will be ignored by the model. Consider removing these intents or add more training examples'
  },
  'unbalanced_data': {
    title: 'Unbalanced Data',
    description: 'Some intents have too many or too few training examples. This might make the model biased towards or against these intents. Consider adding or removing redundant training examples'
  }
}

const theme = getTheme();

const classNames = mergeStyleSets({
  itemCell: [
    getFocusStyle(theme, { inset: -1 }),
    {
      minHeight: 54,
      padding: 10,
      boxSizing: 'border-box',
      borderBottom: `1px solid ${theme.semanticColors.bodyDivider}`,
    }
  ],
})

export const TrainingStatsPage = () => {
  const { demoTrainingFile, setDemoTrainingFile } = useContext(AppContext);

  // component states
  const [ trainingStatsLoading, setTrainingStatsLoading ] = useState(false);

  // data states
  const [ trainingStatsData, setTrainingStatsData ] = useState(null);

  const _fullWidthPlotContainerElement = useRef();

  useEffect(() => {
    if (!demoTrainingFile) {
      setTrainingStatsData(null);
      return;
    }

    const fetchTrainingStatsData = async () => {
      setTrainingStatsLoading(true);

      const resp = await AnalyticsAPI.getTrainingStats({ 
        file: demoTrainingFile,
        sid: socket.id,
      })

      const task_id = resp.data.task_id

      if (task_id) {
        awaitTaskResult(task_id, (data) => {
          console.log(data)
          setTrainingStatsLoading(false);
          setTrainingStatsData(data);
        });
      }
    }

    fetchTrainingStatsData();
  }, [demoTrainingFile])

  const fullWidth = _fullWidthPlotContainerElement.current
    ? _fullWidthPlotContainerElement.current.clientWidth
    : 500;

  return (<div className="ms-Grid" dir="ltr">
  <Stack tokens={{
    childrenGap: 20,
  }}>
    <div>
      <DemoTrainingFileSelector onDemoFileClick={setDemoTrainingFile} />
    </div>

    <div className="ms-Grid-row" ref={_fullWidthPlotContainerElement}>
      {trainingStatsLoading && 
        <Spinner 
          label="Training a simple model on your data. This might take a few minutes..." 
          ariaLive="assertive" 
          labelPosition="left"
          style={{
            justifyContent: 'left'
          }}
        />}
    </div>
      
    {trainingStatsData && <Pivot aria-label="Training Insights">
    
    <PivotItem
      headerText="Overview"
    >
      <div className="ms-Grid-row">
        <div className="ms-Grid-col ms-sm8">
          {trainingStatsData.overall_plot && <div>
            <Plot 
              data={trainingStatsData.overall_plot.data}
              layout={{
                width: fullWidth * .66,
                height: fullWidth * .66 * .75,
                ...trainingStatsData.overall_plot.layout
              }}
              config={{
                responsive: true,
                displaylogo: false,
              }}
            />
          </div>}
        </div>
        <div className="ms-Grid-col ms-sm4">
          {trainingStatsData.stats_overall && <Stack tokens={{
            childrenGap: 20,
          }}>
            <div><Text variant="medium">Model Statistics</Text></div>
            <div>
              <div>
                <Text variant="small">Recall: </Text>
                <Text>{Math.round(trainingStatsData.stats_overall.recall * 100) / 100}</Text>
              </div>
              <div>
                <Text variant="small">Precision: </Text>
                <Text>{Math.round(trainingStatsData.stats_overall.precision * 100) / 100}</Text>
              </div>
              <div>
                <Text variant="small">F1: </Text>
                <Text>{Math.round(trainingStatsData.stats_overall.f1 * 100) / 100}</Text>
              </div>
            </div>
            <div><Text variant="medium">Data Statistics</Text></div>
            <div>
              <div>
                <Text variant="small">Total Intents: </Text>
                <Text>{trainingStatsData.stats_overall.intents_count}</Text> 
              </div>
              <div>
                <Text variant="small">Total Training Examples: </Text>
                <Text>{trainingStatsData.stats_overall.examples_count}</Text> 
              </div>  
            </div>
            <div>
              <div>
                <Text variant="small">Most examples: </Text>
                <Text>{'' + trainingStatsData.stats_overall.max_examples.value}</Text> 
              </div>
              <div>
                <Text variant="small">Least examples: </Text>
                <Text>{'' + trainingStatsData.stats_overall.min_examples.value}</Text> 
              </div>
              <div>
                <Text variant="small">Average examples: </Text>
                <Text>{'' + trainingStatsData.stats_overall.median}</Text> 
              </div>
            </div>
          </Stack>}
        </div>
      </div>

      {trainingStatsData.problems && <div className="ms-Grid-row">
        <Stack tokens={{
          childrenGap: 20,
        }}>
          {trainingStatsData.problems
          .filter(problem => problem.name && problemsData[problem.name])
          .map(problem => <MessageBar 
            key={problem.name}
            messageBarType={MessageBarType.warning}
          >
            <b>{problemsData[problem.name].title + ' '}</b>
            <p>
              {problemsData[problem.name].description}
            </p>
            {problem.intents && <><p>Affected Intents</p>
            <ul>
              {problem.intents.map(problem_intent => <li key={problem_intent}>
                {problem_intent}
              </li>)}
            </ul></>}
          </MessageBar>)}
        </Stack>
      </div>}
    </PivotItem>

    <PivotItem
      headerText="Threshold Levels"
    >
      <div className="ms-Grid-row">
        {trainingStatsData.thresholds_plot && <div>
          <Plot 
            data={trainingStatsData.thresholds_plot.data}
            layout={{
              width: fullWidth * .5,
              height: fullWidth * .5 * .75,
              ...trainingStatsData.thresholds_plot.layout
            }}
            config={{
              responsive: true,
              displaylogo: false,
            }}
          />
        </div>}

        <Text variant="small">Suggested level: <Text variant="smallPlus">{Math.round(trainingStatsData.suggested_threshold * 100)}</Text></Text>
      </div>
    </PivotItem>

    <PivotItem
      headerText="Breakdown by Intents"
    >
      <Stack tokens={{
        childrenGap: 20,
      }}>
      {trainingStatsData.overall_intents_plot && <div>
        <Plot 
          data={trainingStatsData.overall_intents_plot.data}
          layout={{
            width: fullWidth,
            height: fullWidth * .75,
            ...trainingStatsData.overall_intents_plot.layout
          }}
          config={{
            responsive: true,
            displaylogo: false,
          }}
        />
      </div>}

      <div>
      <FocusZone>
        {trainingStatsData.results_intents && <List
        items={trainingStatsData.results_intents
          .sort((a, b) => a.accuracy - b.accuracy)
        }
        onRenderCell={(item, item_idx) => 
          <div key={item_idx} className={classNames.itemCell + ' ms-Grid'} data-is-focusable={true}>
            <div className="ms-Grid-row" style={{ width: '100%' }}>
              <div className="ms-Grid-col ms-sm6">
                <Text variant="mediumPlus">{item.name}</Text>
              </div>
              <div className="ms-Grid-col ms-sm2">
                <div><Text variant="small">Correct</Text></div>
                <div><Text variant="smallPlus">{'' + item.correct}</Text></div>
              </div>
              <div className="ms-Grid-col ms-sm2">
                <div><Text variant="small">Incorrect</Text></div>
                <div><Text variant="smallPlus">{'' + item.incorrect}</Text></div>
              </div>
              <div className="ms-Grid-col ms-sm2">
                <div><Text variant="small">Accuracy</Text></div>
                <div><Text variant="smallPlus">{`${Math.round(item.accuracy * 10000) / 100}%`}</Text></div>
              </div>
            </div>
          </div>}
        />}
      </FocusZone>
      </div></Stack>
    </PivotItem>
    </Pivot>}
  </Stack>
  </div>)
}

export default TrainingStatsPage;
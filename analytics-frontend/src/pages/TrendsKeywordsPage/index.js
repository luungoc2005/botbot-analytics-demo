import React, { useContext, useEffect, useState, useRef, useMemo } from 'react';

import { ChoiceGroup } from 'office-ui-fabric-react/lib/ChoiceGroup';
// import { ProgressIndicator } from 'office-ui-fabric-react/lib/ProgressIndicator';
import { Text } from 'office-ui-fabric-react/lib/Text';
import { Spinner } from 'office-ui-fabric-react/lib/Spinner';
import { Stack } from 'office-ui-fabric-react/lib/Stack';
// import { Label } from 'office-ui-fabric-react/lib/Label';
// import { DetailsList, Selection } from 'office-ui-fabric-react/lib/DetailsList';
import { List } from 'office-ui-fabric-react/lib/List';
import { Checkbox } from 'office-ui-fabric-react/lib/Checkbox';
// import { Callout } from 'office-ui-fabric-react/lib/Callout';
import { DefaultButton } from 'office-ui-fabric-react/lib/Button';
// import { TextField } from 'office-ui-fabric-react/lib/TextField';
import { ComboBox } from 'office-ui-fabric-react/lib/ComboBox';

import { mergeStyleSets, getTheme, normalize } from 'office-ui-fabric-react/lib/Styling';

import { DemoFileSelector } from '../../components/DemoFileSelector';

import Plot from 'react-plotly.js'

import { AppContext } from '../../context';
import { AnalyticsAPI, awaitTaskResult, socket } from '../../api';

const theme = getTheme();

const styles = mergeStyleSets({
  buttonArea: {
    verticalAlign: 'top',
    display: 'inline-block',
    height: 32,
  },
  callout: {
    maxWidth: 300
  },
  container: {
    overflow: 'auto',
    maxHeight: 300,
  },
  containerSegmented: {
    overflow: 'auto',
    maxHeight: 300,
    selectors: {
      '.ms-List-cell': {
        borderLeft: '3px solid ' + theme.palette.themePrimary,
        height: 50,
        lineHeight: 50,
      },
      '.ms-List-cell:nth-child(odd)': {
        background: theme.palette.neutralLighter
      },
      '.ms-List-cell:nth-child(even)': {
      }
    }
  },
  itemContent: [
    theme.fonts.medium,
    normalize,
    {
      position: 'relative',
      display: 'block',
      paddingLeft: 27,
      paddingTop: 10,
      paddingBottom: 10,
      overflow: 'hidden',
      wordBreak: 'keep-all',
      textOverflow: 'ellipsis',
    }
  ]
})

export const TrendsKeywordsPage = () => {
  const { demoFile, setDemoFile } = useContext(AppContext);
  const [ wordSearchInput, setWordSearchInput ] = useState('');
  const [ selectedWords, setSelectedWords ] = useState([]);
  const [ selectedPeriod, setSelectedPeriod ] = useState('D');

  // data states
  const [ intentsData, setIntentsData ] = useState([]);
  const [ topWordsData, setTopWordsData ] = useState([]);
  const [ wordsTrendData, setWordsTrendData ] = useState(null);
  const [ similarWordsData, setSimilarWordsData ] = useState([]);

  const _wordsChartContainerElement = useRef();
  const _pendingTasks = useRef([])

  useEffect(() => {
    if (!demoFile) {
      setIntentsData([]);
      return;
    }

    const fetchIntentsData = async () => {
      const resp = await AnalyticsAPI.getIntentsList({ file: demoFile })
      setIntentsData(resp.data.map(name => ({ name, checked: true })))
    }
    fetchIntentsData();
  }, [demoFile])

  useEffect(() => {
    if (!demoFile) {
      setTopWordsData([]);
      return;
    }

    const fetchTopWordsData = async () => {
      const resp = await AnalyticsAPI.getTopWords({
        file: demoFile,
        only: intentsData
          .map(item => item.name)
          .join(','),
        top_n: 50,
        sid: socket.id,
      })
      setTopWordsData(resp.data);
    }
    fetchTopWordsData();
  }, [demoFile, intentsData])

  useEffect(() => {
    if (!demoFile) {
      setWordsTrendData(null);
      return;
    }
    const words = selectedWords
      .filter(item => item.checked)
      .map(item => item.text);

    if (words.length === 0) {
      return;
    }
    
    const fetchWordsTrendData = async () => {
      const resp = await AnalyticsAPI.getWordsTrend({
        file: demoFile,
        period: selectedPeriod,
        words: words.join(','),
      })
      setWordsTrendData(resp.data);
    }
    fetchWordsTrendData();
  }, [demoFile, selectedWords, selectedPeriod])

  useEffect(() => {
    if (!demoFile) {
      setWordsTrendData(null);
      return;
    }
    const words = selectedWords
      .filter(item => item.checked)
      .map(item => item.text);

    if (words.length === 0) {
      return;
    }
     
    _pendingTasks.current.forEach(cancelTask => cancelTask())
    _pendingTasks.current = []

    let newData = words.map(word => ({ word, task_id: null }));

    const fetchSimilarWordsData = async () => {
      setSimilarWordsData(newData);

      words.forEach(async (word, word_idx) => {
        const resp = await AnalyticsAPI.getSimilarWords({
          file: demoFile,
          word,
          top_n: 5,
          sid: socket.id
        })
        const task_id = resp.data.task_id
        if (task_id) {
          newData[word_idx].task_id = task_id;
          const cancelTask = awaitTaskResult(task_id, (task_resp) => {
            newData[word_idx].data = task_resp;
            console.log(task_id, task_resp, newData, words.length);
            console.log('filtered', newData.filter(item => item && item.data));
            if (newData.filter(item => item && item.data).length === words.length) {
              setSimilarWordsData([...newData]);
            }
          });
          _pendingTasks.current.push(cancelTask)
        }
      })
    }
    fetchSimilarWordsData();
  }, [demoFile, selectedWords])

  return (<Stack tokens={{
    childrenGap: 20,
  }}>
    <div>
      <DemoFileSelector onDemoFileClick={setDemoFile} />
    </div>
    
    {demoFile && <>
    
    <Text variant="large">Keywords Interest over Time</Text>

    <div className="ms-Grid" dir="ltr">
    <div className="ms-Grid-row">
      <div className="ms-Grid-col ms-sm5">
      <ComboBox
        autoComplete="on"
        // buttonIconProps={{ iconName: 'Search' }}
        options={topWordsData.map(item => ({ key: item.word, text: item.word }))}
        onChange={(e, option, index, value) => {
          if (option) {
            setWordSearchInput(option.key)
          }
          else {
            setWordSearchInput(value);
          }
        }}
        // defaultSelectedKey="C"
        selectedKey={wordSearchInput}
        allowFreeform={true}
        // useComboBoxAsMenuWidth={true}
      />
      </div>

      <div className="ms-Grid-col ms-sm3">
        <DefaultButton 
          text='Add to comparison' 
          iconProps={{ iconName: 'Add' }}
          onClick={() => {
            if (wordSearchInput &&
              !selectedWords.find(item => item.text.toLocaleLowerCase() === wordSearchInput.toLocaleLowerCase())) {
              setSelectedWords([...selectedWords, {
                text: wordSearchInput,
                checked: true,
              }])
            }
          }}
        />
      </div>
      
      <div className="ms-Grid-col ms-sm4">
      <ChoiceGroup
        defaultSelectedKey={selectedPeriod}
        options={[
          {
            key: 'D',
            iconProps: { iconName: 'CalendarDay' },
            text: 'Day',
          },
          {
            key: 'M',
            iconProps: { iconName: 'Calendar' },
            text: 'Month',
          }
        ]}
        onChange={(e, option) => setSelectedPeriod(option.key)}
      />
      </div>
    </div>
    </div>

    <div className="ms-Grid" dir="ltr">
    <div className="ms-Grid-row">
      <div className="ms-Grid-col ms-sm4">
        <div className={styles.containerSegmented} data-is-scrollable={true}>
          <List 
            items={selectedWords}
            onRenderCell={(item, index) => <div data-is-focusable={true} key={index}>
              <div className={styles.itemContent}>
                <Checkbox 
                  label={item.text} 
                  defaultChecked={item.checked}
                  onChange={() => setSelectedWords(selectedWords.map(
                    word => word.text === item.text ? {
                      ...word,
                      checked: !item.checked
                    } : word
                  ))}
                />
              </div>
            </div>}
          />
        </div>
      </div>
    </div>

    <div className="ms-Grid-row">
      <div className="ms-Grid-col ms-sm12" ref={_wordsChartContainerElement}>
        {wordsTrendData && <Plot
          data={wordsTrendData.data}
          layout={{
            width: _wordsChartContainerElement.current ? _wordsChartContainerElement.current.clientWidth : 400,
            height: _wordsChartContainerElement.current ? _wordsChartContainerElement.current.clientWidth / 2 : 250,
            ...wordsTrendData.layout
          }}
          config={{
            responsive: true,
            displaylogo: false,
          }}
        />}
      </div>
    </div>
    </div>

    <div className="ms-Grid-row">
      {similarWordsData && similarWordsData.filter(item => item)
        .map((item, idx) => <div className="ms-Grid-col ms-sm4" key={idx}>
          <Stack tokens={{
            childrenGap: 20,
          }}>
            <Text style={{ paddingTop: 20 }}>
              Most related to "{item.word}"
            </Text>
            {item.data
            ? <List
                items={item.data}
                onRenderCell={(word, word_idx) => <div 
                  data-is-focusable={true} 
                  key={word_idx}
                >
                  <div className={styles.itemContent}>
                    #{word_idx + 1}: {word}
                  </div>
                </div>}
              />
            : <Spinner 
                label="Loading..." 
                ariaLive="assertive"
              />}
            </Stack>
        </div>)}
    </div>

    </>}
  </Stack>)
}

export default TrendsKeywordsPage;
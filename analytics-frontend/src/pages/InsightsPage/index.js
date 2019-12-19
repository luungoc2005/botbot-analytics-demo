import React, { useContext, useEffect, useState, useRef } from 'react';

// import { ChoiceGroup } from 'office-ui-fabric-react/lib/ChoiceGroup';
// import { ProgressIndicator } from 'office-ui-fabric-react/lib/ProgressIndicator';
import { Text } from 'office-ui-fabric-react/lib/Text';
// import { Spinner } from 'office-ui-fabric-react/lib/Spinner';
import { Stack } from 'office-ui-fabric-react/lib/Stack';
// import { Label } from 'office-ui-fabric-react/lib/Label';
// import { DetailsList, Selection } from 'office-ui-fabric-react/lib/DetailsList';
import { List } from 'office-ui-fabric-react/lib/List';
import { Checkbox } from 'office-ui-fabric-react/lib/Checkbox';
import { Callout } from 'office-ui-fabric-react/lib/Callout';
import { ActionButton } from 'office-ui-fabric-react/lib/Button';

import { mergeStyleSets, getTheme, normalize } from 'office-ui-fabric-react/lib/Styling';

import { DemoFileSelector } from '../../components/DemoFileSelector';

import Plot from 'react-plotly.js'

import { AppContext } from '../../context';
import { AnalyticsAPI } from '../../api';

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

export const InsightsPage = () => {
  const { demoFile, setDemoFile } = useContext(AppContext);
  const [ isCalloutVisible, setIsCalloutVisible ] = useState(false);
  const [ intentsData, setIntentsData ] = useState([]);
  const [ topIntentsData, setTopIntentsData ] = useState([]);
  const [ topWordsData, setTopWordsData ] = useState([]);
  const _menuButtonElement = useRef();

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
      setTopIntentsData([]);
      return;
    }

    const fetchTopIntentsData = async () => {
      const resp = await AnalyticsAPI.getTopIntents({
        file: demoFile,
        only: intentsData
          .filter(item => item.checked)
          .map(item => item.name)
          .join(','),
        top_n: 10,
      })
      setTopIntentsData(resp.data);
    }
    fetchTopIntentsData();
  }, [demoFile, intentsData])

  useEffect(() => {
    if (!demoFile) {
      setTopWordsData([]);
      return;
    }

    const fetchTopWordsData = async () => {
      const resp = await AnalyticsAPI.getTopWords({
        file: demoFile,
        only: intentsData
          .filter(item => item.checked)
          .map(item => item.name)
          .join(','),
        top_n: 50,
      })
      setTopWordsData(resp.data);
    }
    fetchTopWordsData();
  }, [demoFile, intentsData])
  const toggleIsCalloutVisible = () => setIsCalloutVisible(!isCalloutVisible);

  return (<Stack tokens={{
    childrenGap: 20,
  }}>
    <div>
      <DemoFileSelector onDemoFileClick={setDemoFile} />
    </div>
    
    {demoFile && intentsData
    ? <>
      <div style={{ textAlign: 'right', width: '100%' }}>
        <div className={styles.buttonArea} ref={_menuButtonElement}>
          <ActionButton
            iconProps={{ iconName: 'Filter' }}
            onClick={toggleIsCalloutVisible}
          >
            Filter Intents
          </ActionButton>
        </div>
        {isCalloutVisible 
        ? (<div>
            <Callout
              className={styles.callout}
              gapSpace={0}
              target={_menuButtonElement.current}
              onDismiss={() => setIsCalloutVisible(false)}
              setInitialFocus={true}
            >
              <div className={styles.containerSegmented} data-is-scrollable={true}>
                <List 
                  items={intentsData}
                  onRenderCell={(item, index) => <div data-is-focusable={true} key={index}>
                    <div className={styles.itemContent}>
                      <Checkbox 
                        label={item.name} 
                        defaultChecked={item.checked}
                        onChange={() => setIntentsData(intentsData.map(
                          intent => intent.name === item.name ? {
                            ...intent,
                            checked: !intent.checked
                          } : intent
                        ))}
                      />
                    </div>
                  </div>}
                />
              </div>
            </Callout>
          </div>)
        : null}
      </div>

      <div className="ms-Grid" dir="ltr">
      <div className="ms-Grid-row">
        <div className="ms-Grid-col ms-sm6">
          <Text variant="large">Top Intents</Text>
          <div className={styles.container} data-is-scrollable={true}>
            <List 
              items={topIntentsData}
              onRenderCell={(item, index) => <div data-is-focusable={true} key={index}>
                <div className={styles.itemContent}>
                  #{index + 1}: {item.name}
                  <Text variant="small"> ({item.count})</Text>
                </div>
              </div>}
            />
          </div>
        </div>

      <div className="ms-Grid-col ms-sm6">
        <Text variant="large">Top Words</Text>
        <div className={styles.container} data-is-scrollable={true}>
          <List 
            items={topWordsData.slice(0, 10)}
            onRenderCell={(item, index) => <div data-is-focusable={true} key={index}>
              <div className={styles.itemContent}>
                #{index + 1}: {item.word}
                <Text variant="small"> ({item.count})</Text>
              </div>
            </div>}
          />
        </div>
      </div>

    </div>
    </div>
    </>
    : <></>}
  </Stack>)
}

export default InsightsPage
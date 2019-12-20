import React, { useContext, useEffect, useState, useRef, useMemo } from 'react';

import { ChoiceGroup } from 'office-ui-fabric-react/lib/ChoiceGroup';
// import { ProgressIndicator } from 'office-ui-fabric-react/lib/ProgressIndicator';
import { Text } from 'office-ui-fabric-react/lib/Text';
// import { Spinner } from 'office-ui-fabric-react/lib/Spinner';
import { Stack } from 'office-ui-fabric-react/lib/Stack';
// import { Label } from 'office-ui-fabric-react/lib/Label';
// import { DetailsList, Selection } from 'office-ui-fabric-react/lib/DetailsList';
import { List } from 'office-ui-fabric-react/lib/List';
import { Checkbox } from 'office-ui-fabric-react/lib/Checkbox';
import { Callout } from 'office-ui-fabric-react/lib/Callout';
import { DefaultButton } from 'office-ui-fabric-react/lib/Button';
// import { TextField } from 'office-ui-fabric-react/lib/TextField';
import { ComboBox } from 'office-ui-fabric-react/lib/ComboBox';

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

export const TrendsIntentsPage = () => {
  const { demoFile, setDemoFile } = useContext(AppContext);
  const [ intentSearchInput, setIntentSearchInput ] = useState([]);
  const [ selectedPeriod, setSelectedPeriod ] = useState('D');

  // data states
  const [ intentsData, setIntentsData ] = useState([]);
  const [ intentsTrendData, setIntentsTrendData ] = useState(null);

  const _intentsChartContainerElement = useRef();

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
      setIntentsTrendData(null);
      return;
    }
    const intents = intentsData
      .map(item => item.name)
      .filter(item => intentSearchInput.indexOf(item) > -1)

    if (intents.length === 0) {
      return;
    }
    
    const fetchIntentsTrendData = async () => {
      const resp = await AnalyticsAPI.getIntentsTrend({
        file: demoFile,
        period: selectedPeriod,
        intents: intents.join(','),
      })
      setIntentsTrendData(resp.data);
    }
    fetchIntentsTrendData();
  }, [demoFile, intentSearchInput, selectedPeriod])

  return (<Stack tokens={{
    childrenGap: 20,
  }}>
    <div>
      <DemoFileSelector onDemoFileClick={setDemoFile} />
    </div>
    
    {demoFile && <>
    
    <Text variant="large">Intents Interest over Time</Text>

    <div className="ms-Grid" dir="ltr">
    <div className="ms-Grid-row">
      <div className="ms-Grid-col ms-sm8">
      <ComboBox
        autoComplete="on"
        multiSelect
        // buttonIconProps={{ iconName: 'Search' }}
        options={intentsData.map(item => ({ key: item.name, text: item.name }))}
        onChange={(e, option, index, value) => {
          if (option) {
            let selectedKeys = [...intentSearchInput]; // modify a copy
            const index = selectedKeys.indexOf(option.key);
            if (option.selected && index < 0) {
              selectedKeys.push(option.key);
            }
            else {
              selectedKeys.splice(index, 1);
            }
            setIntentSearchInput(selectedKeys);
          }
        }}
        // defaultSelectedKey="C"
        selectedKey={intentSearchInput}
        // allowFreeform={true}
        // useComboBoxAsMenuWidth={true}
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

    <div className="ms-Grid-row">
      <div className="ms-Grid-col ms-sm12" ref={_intentsChartContainerElement}>
        {intentsTrendData && <Plot
          data={intentsTrendData.data}
          layout={{
            width: _intentsChartContainerElement.current ? _intentsChartContainerElement.current.clientWidth : 400,
            height: _intentsChartContainerElement.current ? _intentsChartContainerElement.current.clientWidth / 2 : 250,
            ...intentsTrendData.layout
          }}
          config={{
            responsive: true
          }}
        />}
      </div>
    </div>

    </>}
  </Stack>)
}

export default TrendsIntentsPage;
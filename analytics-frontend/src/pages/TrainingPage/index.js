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

import { DemoTrainingFileSelector } from '../../components/DemoTrainingFileSelector';

import Plot from 'react-plotly.js'

import { AppContext } from '../../context';
import { AnalyticsAPI } from '../../api';

export const TrainingPage = () => {
  const { demoTrainingFile, setDemoTrainingFile } = useContext(AppContext);

  return (<Stack tokens={{
    childrenGap: 20,
  }}>
    <div>
      <DemoTrainingFileSelector onDemoFileClick={setDemoTrainingFile} />
    </div>
    
    
  </Stack>)
}

export default TrainingPage;
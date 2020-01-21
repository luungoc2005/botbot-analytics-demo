import React, { useContext } from 'react';

import { AppContext } from '../../context';

import { getTheme, mergeStyleSets } from 'office-ui-fabric-react/lib/Styling';
import { Text } from 'office-ui-fabric-react/lib/Text';
import { FocusZone } from 'office-ui-fabric-react/lib/FocusZone';
import { List } from 'office-ui-fabric-react/lib/List';
import { CompoundButton } from 'office-ui-fabric-react/lib/Button';
import { TooltipHost, TooltipOverflowMode } from 'office-ui-fabric-react/lib/Tooltip';

const { palette, fonts } = getTheme();

const classNames = mergeStyleSets({
  listGridExample: {
    overflow: 'hidden',
    fontSize: 0,
    position: 'relative',    
  },
  listGridExampleTile: {
    textAlign: 'center',
    outline: 'none',
    position: 'relative',
    float: 'left',
    marginRight: 20,
    marginBottom: 20,
    padding: 0,
    // background: palette.neutralLighter,
    selectors: {
      'focus:after': {
        content: '',
        position: 'absolute',
        left: 2,
        right: 2,
        top: 2,
        bottom: 2,
        boxSizing: 'border-box',
        border: `1px solid ${palette.white}`
      },
      ':active': {
        padding: 0,
      }
    }
  },
  listGridExampleSizer: {
    paddingBottom: '100%'
  },
  listGridExamplePadder: {
    position: 'absolute',
    left: 2,
    top: 2,
    right: 2,
    bottom: 2
  },
  listGridExampleLabel: {
    background: 'rgba(0, 0, 0, 0.3)',
    color: '#FFFFFF',
    position: 'absolute',
    padding: 10,
    bottom: 0,
    left: 0,
    width: '100%',
    fontSize: fonts.tiny.fontSize,
    boxSizing: 'border-box',
    userSelect: 'none',
    textOverflow: 'ellipsis',
    wordBreak: 'keep-all',
    overflow: 'hidden',
  },
  listGridExampleImage: {
    position: 'absolute',
    top: 0,
    left: 0,
    width: '100%'
  }
});

export const DemoTrainingFileSelector = ({ onDemoFileClick = null }) => {
  const { demoTrainingFile, demoTrainingList } = useContext(AppContext);
  return <>
    <Text>Please select a demo training file</Text>
    <FocusZone style={{ paddingTop: 20, paddingBottom: 20 }}>
      <List
        className={classNames.listGridExample}
        items={demoTrainingList}
        getItemCountForPage={() => 5}
        getPageHeight={() => window.innerHeight}
        renderedWindowsAhead={10}
        onRenderCell={(item, index) => <CompoundButton
          key={index}
          toggle
          checked={Boolean(demoTrainingFile === item.name)}
          style={{ padding: 0 }}
          onClick={() => onDemoFileClick && onDemoFileClick(item.name)}
          className={classNames.listGridExampleTile}
          data-is-focusable={true}
          style={{
            width: 120,
            height: 100,
          }}
        >
          <div className={classNames.listGridExampleSizer}>
            <div className={classNames.listGridExamplePadder}>
              <img className={classNames.listGridExampleImage} />
              <span className={classNames.listGridExampleLabel}>
                <TooltipHost
                  overflowMode={TooltipOverflowMode.Parent}
                  content={item.name}
                >
                  <Text variant="tiny">
                    {item.name}
                  </Text>
                </TooltipHost>
              </span>
            </div>
          </div>
        </CompoundButton>}
      ></List>
    </FocusZone>
  </>
}

export default DemoTrainingFileSelector;
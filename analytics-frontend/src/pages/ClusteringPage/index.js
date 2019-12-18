import React, { useContext } from 'react';

import { AppContext } from '../../context';

import { getTheme, mergeStyleSets } from 'office-ui-fabric-react/lib/Styling';
import { Text } from 'office-ui-fabric-react/lib/Text';
import { FocusZone } from 'office-ui-fabric-react/lib/FocusZone';
import { List } from 'office-ui-fabric-react/lib/List';
import { CompoundButton } from 'office-ui-fabric-react/lib/Button';

const { palette, fonts } = getTheme();

const classNames = mergeStyleSets({
  listGridExample: {
    overflow: 'hidden',
    fontSize: 0,
    position: 'relative'
  },
  listGridExampleTile: {
    textAlign: 'center',
    outline: 'none',
    position: 'relative',
    float: 'left',
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
    fontSize: fonts.small.fontSize,
    boxSizing: 'border-box',
    userSelect: 'none',
  },
  listGridExampleImage: {
    position: 'absolute',
    top: 0,
    left: 0,
    width: '100%'
  }
});

export const ClusteringPage = () => {
  const { demoList } = useContext(AppContext);
  return <>
    <Text>Please select a demo history file</Text>
    <FocusZone style={{ paddingTop: 20 }}>
      <List
        className={classNames.listGridExample}
        items={demoList}
        getItemCountForPage={() => 5}
        getPageHeight={() => window.innerHeight}
        renderedWindowsAhead={10}
        onRenderCell={(item, index) => <CompoundButton style={{ padding: 0 }}>
          <div
            key={index}
            className={classNames.listGridExampleTile}
            data-is-focusable={true}
            style={{
              width: 250
            }}
          >
            <div className={classNames.listGridExampleSizer}>
              <div className={classNames.listGridExamplePadder}>
                <img className={classNames.listGridExampleImage} />
                <span className={classNames.listGridExampleLabel}>
                  <Text>
                    {item}
                  </Text>
                </span>
              </div>
            </div>
          </div>
        </CompoundButton>}
      ></List>
    </FocusZone>
  </>
}

export default ClusteringPage;
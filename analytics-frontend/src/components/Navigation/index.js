import React from 'react';
import {
  Nav
} from 'office-ui-fabric-react/lib/Nav';

export const Navigation = () => {
  return <Nav
    styles={{
      root: {
        width: '100%',
        height: '100%',
        boxSizing: 'border-box',
        border: '1px solid #eee',
        overflowY: 'auto'
      }
    }}
    groups={[
      {
        name: 'Content Analytics',
        links: [
          {
            name: 'Clustering',
            url: '/clustering',
          },
          {
            name: 'Insights',
            url: '/insights',
          },
          {
            name: 'Trends',
            // url: '/trends',
            links: [
              {
                name: 'Keywords',
                url: '/trends_keywords'
              },
              {
                name: 'Intents',
                url: '/trends_intents'
              }
            ]
          },
        ]
      },
      {
        name: 'Training',
        links: [
          {
            name: 'Training Data',
            url: '/training',
          }
        ]
      }
    ]}
  />
}

export default Navigation;
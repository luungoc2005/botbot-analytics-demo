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
        name: 'Home',
        links: [
          {
            name: 'Clustering',
            url: '/clustering',
          }
        ]
      }
    ]}
  />
}

export default Navigation;
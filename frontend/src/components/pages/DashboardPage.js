import React from 'react';
import { Typography, Grid, Tooltip } from '@mui/material';
import OurPieChart from '../OurPieChart';
import OurBarChart from '../OurBarChart';
import ResultsGrid from '../ResultsGrid';

const DashboardPage = (props) => {
  const dummy_rows=props.analyze
  // const dummy_rows = [
  //   { id: 1, class: 'Fake News', accuracy: 59.1, content: 'first tweet' },
  //   { id: 2, class: 'Fake News', accuracy: 68.4, content: 'second tweet' },
  //   { id: 3, class: 'True News', accuracy: 75.0, content: 'third tweet' },
  //   { id: 4, class: 'Fake News', accuracy: 89.7, content: 'blalalalg' },
  //   {
  //     id: 5,
  //     class: 'True News',
  //     accuracy: 78.4,
  //     content: 'fjgkdjlofjfsgdgdhdg',
  //   },
  // ];

  const styles = {
    mainContainer: {
      my: '5em',
    },
  };
  return (
    <Grid
      container
      direction="column"
      sx={styles.mainContainer}
      alignItems="center"
    >
      <Typography
        variant="h3"
        justifyContent="center"
        align="center"
        sx={{ mb: 7 }}
      >
        Dashboard
      </Typography>
      <Grid container justifyContent="center">
        <Grid item>
          <Tooltip
            title="Average accuracy for each class"
            placement="top"
            componentsProps={{
              tooltip: {
                sx: {
                  fontSize: 16,
                  width: '100%',
                },
              },
            }}
          >
            <span>
              <OurBarChart tweets={dummy_rows} />
            </span>
          </Tooltip>
        </Grid>
        <Grid item>
          <Tooltip
            title="Number of tweets for each class"
            placement="top"
            componentsProps={{
              tooltip: {
                sx: {
                  fontSize: 16,
                  width: '100%',
                },
              },
            }}
          >
            <span>
              <OurPieChart tweets={dummy_rows} />
            </span>
          </Tooltip>
        </Grid>
      </Grid>
      <ResultsGrid tweets={dummy_rows} />
    </Grid>
  );
};

export default DashboardPage;

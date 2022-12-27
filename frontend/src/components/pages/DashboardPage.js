import React from 'react';
import { Typography, Grid, Tooltip } from '@mui/material';
import OurPieChart from '../OurPieChart';
import OurBarChart from '../OurBarChart';
import ResultsGrid from '../ResultsGrid';

const DashboardPage = (props) => {
  console.log(props.analyze)
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
              <OurBarChart tweets={props.analyze} />
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
              <OurPieChart tweets={props.analyze} />
            </span>
          </Tooltip>
        </Grid>
      </Grid>
      <ResultsGrid tweets={props.analyze} />
    </Grid>
  );
};

export default DashboardPage;

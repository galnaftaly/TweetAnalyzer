import React from 'react';
import { Typography, Grid, Tooltip, Button } from '@mui/material';
import OurPieChart from '../OurPieChart';
import OurBarChart from '../OurBarChart';
import ResultsGrid from '../ResultsGrid';
import { Link } from 'react-router-dom';

const DashboardPage = (props) => {
  const styles = {
    mainContainer: {
      my: '5em',
    },
    button: (theme) => ({
      height: 60,
      width: 250,
      fontSize: 22,
      m: '25px',
      borderRadius: 3,
      '&:hover': {
        backgroundColor: theme.palette.primary.light,
      },
    }),
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
      {props.analyze.length === 0 ? (
        <Grid container justifyContent="center" alignItems="center" direction="column">
          <Grid item>
            <Typography variant="h4" sx={{ color: 'black' }}>
              You must insert tweets before navigate to dashboard page.
            </Typography>
          </Grid>
          <Grid display="flex" item>
            <Button
              component={Link}
              to="/tweets"
              variant="contained"
              sx={styles.button}
              onClick={() => props.setValue(4)}
              align="center"
            >
              Insert Tweets
            </Button>
          </Grid>
        </Grid>
      ) : (
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
          <ResultsGrid tweets={props.analyze} />
        </Grid>
      )}
    </Grid>
  );
};

export default DashboardPage;

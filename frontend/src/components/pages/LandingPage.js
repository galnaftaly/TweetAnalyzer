import React from 'react';
import { Grid, Button, Box, Typography } from '@mui/material';
import landing from '../../assests/landing.jpg';
import { Link } from 'react-router-dom';

const LandingPage = (props) => {
  const styles = {
    landingImg: {
      height: '100%',
      width: '100%',
      borderRadius: 5,
    },
    mainContainer: {
      mt: '5em',
    },
    textContainer: {
      ml: '1em',
    },
    button: (theme) => ({
      height: 60,
      width: 200,
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
      minHeight="750px"
    >
      <Grid item>
        <Grid
          container
          direction="row"
          justifyContent="flex-end"
          alignItems="center"
        >
          <Grid sm item sx={styles.textContainer}>
            <Typography variant="h2" align="center">
              Fake News Detection
              <br />
              And Authorship Identification
            </Typography>
            <Grid container justifyContent="center" sx={styles.buttonContainer}>
              <Grid display="flex" item>
                <Button
                  component={Link}
                  to="/about"
                  variant="contained"
                  sx={styles.button}
                  onClick={() => props.setValue(4)}
                >
                  Learn More
                </Button>
              </Grid>
              <Grid item></Grid>
            </Grid>
          </Grid>
          <Grid sm display="flex" item>
            <Box
              component="img"
              alt="landing image"
              src={landing}
              sx={styles.landingImg}
            />
          </Grid>
        </Grid>
      </Grid>
    </Grid>
  );
};

export default LandingPage;

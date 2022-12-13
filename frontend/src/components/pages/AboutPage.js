import React, { useState } from 'react';
import {
  Grid,
  Typography,
  Button,
  Avatar,
  Card,
  CardContent,
} from '@mui/material';
import gal from '../../assests/gal.jpeg';
import aviv from '../../assests/aviv.jpeg';
import ModelDialog from '../ModelDialog';

const AboutPage = () => {
  const [open, setOpen] = useState(false);

  const handleClose = () => {
    setOpen(false);
  };

  const handleOpen = () => {
    setOpen(true);
  };

  const styles = {
    text: {
      fontStyle: 'italic',
      fontWeight: 300,
      fontSize: '1.5rem',
      maxWidth: '70em',
      lineHeight: 1.6,
      my: '1em',
    },
    mainContainer: {
      m: '5em',
    },
    button: (theme) => ({
      fontSize: 22,
      my: '15px',
      borderRadius: 3,
      '&:hover': {
        backgroundColor: theme.palette.primary.light,
      },
    }),
    card: (theme) => ({
      position: 'absolute',
      boxShadow: theme.shadows,
      borderRadius: 15,
      padding: '1em',
      mx: 5,
      border: 2,
      borderColor: theme.palette.common.blue,
    }),
  };

  const TeamCard = () => {
    return (
      <Grid
        container
        sx={{ height: '45em' }}
        alignItems="center"
        justifyContent="center"
      >
        <Card sx={styles.card}>
          <CardContent>
            <Grid item container direction="column" justifyContent="center">
              <Grid item>
                <Typography align="center" variant="h4" sx={{ mb: '0.5em' }}>
                  Team
                </Typography>
              </Grid>
              <Grid item container direction="row" justifyContent="center">
                <Grid item align="center" sx={{ mx: '2em' }}>
                  <Avatar
                    alt="Gal Naftaly"
                    src={gal}
                    sx={{ width: 200, height: 200, m: 4 }}
                  />
                  <Typography align="center" variant="h5">
                    Gal Naftaly
                  </Typography>
                  <Typography align="center" variant="body1" sx={{fontSize: 18}}>
                    <br />
                    gal.naftaly@e.braude.ac.il
                    <br />
                    Information Systems Engineering Student
                  </Typography>
                </Grid>
                <Grid item align="center" sx={{ mx: '2em' }}>
                  <Avatar
                    alt="Aviv Meir"
                    src={aviv}
                    sx={{ width: 200, height: 200, m: 4 }}
                  />
                  <Typography align="center" variant="h5">
                    Aviv Meir
                  </Typography>
                  <Typography align="center" variant="body1" sx={{fontSize: 18}}>
                    <br />
                    aviv.meir@e.braude.ac.il
                    <br />
                    Information Systems Engineering Student
                  </Typography>
                </Grid>
              </Grid>
            </Grid>
          </CardContent>
        </Card>
      </Grid>
    );
  };

  const Text = () => {
    return (
      <Typography align="center" variant="h4" sx={styles.text}>
        The application used for two tasks: fake news detection and authorship
        identification on inserted tweets and outputs the precentege of
        possibility.
        <br />
        For the fake news detection task the precentege represents if the tweet
        content is true news or fake news.
        <br />
        For the authorship identification task the precentege represents if the
        tweet has been written by human or bot by writing style analysis.
        <br />
        You can upload single tweet or some tweets at the same time.
        <br />
        The application architecture based on BGSRD model.
      </Typography>
    );
  };

  return (
    <Grid
      container
      direction="column"
      justifyContent="center"
      alignItems="center"
    >
      <Grid item>
        <Typography variant="h3" align="center" sx={{ m: 2 }}>
          About Us
        </Typography>
        <Text />
      </Grid>
      <Grid item container justifyContent="center">
        <Button variant="contained" sx={styles.button} onClick={handleOpen}>
          See Model Architecture
        </Button>
      </Grid>
      <ModelDialog open={open} onClose={handleClose} />
      <Grid item>
        <TeamCard />
      </Grid>
    </Grid>
  );
};

export default AboutPage;

import React, { useEffect, useState } from 'react';
import { Typography, Grid } from '@mui/material';
import TrialTable from '../TrialTable';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';

const TweetsPage = (props) => {
  const [fetch, setFetch] = useState(false);
  const navigate = useNavigate();

  const fetchTweets = () => {
    axios
      .post(`http://127.0.0.1:8000/predict`, { data: props.tweetTable })
      .then((res) => {
        props.setAnalyze(res.data);
      })
      .then(() => navigate('/dashboard'))
      .catch((error) => {
        console.log(error);
      });
  };

  useEffect(() => {
    if (fetch === true) {
      fetchTweets();
    }
  }, [fetch]);


  return (
    <Grid container direction="column" alignItems="center">
      <Typography
        variant="h3"
        justifyContent="center"
        align="center"
        alignItems="center"
        sx={{ m: 2 }}
      >
        Insert Tweets
      </Typography>
      <TrialTable setTweetTable={props.setTweetTable} setFetch={setFetch} />
    </Grid>
  );
};

export default TweetsPage;

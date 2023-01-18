import React, { useEffect, useState } from 'react';
import { Typography, Grid, CircularProgress, Box } from '@mui/material';
import TrialTable from '../TrialTable';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import ChooseTask from '../ChooseTask';
import BookTable from '../BookTable';

const TweetsPage = (props) => {
  const [fetch, setFetch] = useState(false);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const fetchTweets = () => {
    setLoading(true);
    axios
      .post(`http://127.0.0.1:8000/predict?dataset=` + props.task, {
        data: props.tweetTable,
      })
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

  function TaskToChoose(props) {
    const isTaskChoosen =
      props.task === '' || props.task === undefined ? false : true;
    if (isTaskChoosen && props.task === 'twitter') {
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
          <TrialTable
            task={props.task}
            setTweetTable={props.setTweetTable}
            setFetch={setFetch}
            tweets={props.tweets}
            setTweets={props.setTweets}
          />
          <Grid item>
            {loading && (
              <Box sx={{ m: 2 }}>
                <CircularProgress />
              </Box>
            )}
          </Grid>
        </Grid>
      );
    }
    if (isTaskChoosen && props.task === 'MR') {
      return (
        <Grid container direction="column" alignItems="center">
          <Typography
            variant="h3"
            justifyContent="center"
            align="center"
            alignItems="center"
            sx={{ m: 2 }}
          >
            Insert Review
          </Typography>
          <TrialTable
            task={props.task}
            setTweetTable={props.setTweetTable}
            setFetch={setFetch}
            tweets={props.tweets}
            setTweets={props.setTweets}
          />
          <Grid item>
            {loading && (
              <Box sx={{ m: 2 }}>
                <CircularProgress />
              </Box>
            )}
          </Grid>
        </Grid>
      );
    }

    if (isTaskChoosen && props.task === 'shakespeare') {
      return (
        <Grid container direction="column" alignItems="center">
          <Typography
            variant="h3"
            justifyContent="center"
            align="center"
            alignItems="center"
            sx={{ m: 2 }}
          >
            Insert Book
          </Typography>
          <BookTable
            setTweetTable={props.setTweetTable}
            setFetch={setFetch}
            books={props.tweets}
            setBooks={props.setTweets}
          />
          <Grid item>
            {loading && (
              <Box sx={{ m: 2 }}>
                <CircularProgress />
              </Box>
            )}
          </Grid>
        </Grid>
      );
    }
    if (!isTaskChoosen) {
      return (
        <ChooseTask
          task={props.task}
          setTask={props.setTask}
          setDataset={props.setDataset}
        />
      );
    }
  }

  return (
    <TaskToChoose
      task={props.task}
      setTweetTable={props.setTweetTable}
      setTask={props.setTask}
      tweets={props.tweets}
      setTweets={props.setTweets}
      setDataset={props.setDataset}
    />
  );
};

export default TweetsPage;

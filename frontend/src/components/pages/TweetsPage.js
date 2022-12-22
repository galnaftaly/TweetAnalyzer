
import React, { useEffect } from "react";
import { Typography, Grid } from "@mui/material";
import TrialTable from "../TrialTable";
import axios from 'axios';


const TweetsPage = (props) => {

  //   const fetchTweets = async()=>{
  //     const data = props.tweetTable
  //     console.log(JSON.stringify(data))
  //     const requestOptions = {
  //       method: "POST",
  //       headers: { "Content-Type": "application/json" },
  //       body: JSON.stringify(data)
  //     };
  //     fetch("http://127.0.0.1:8000/tweets/", requestOptions)
  //       .then(response => response.json())
  //       .then(
  //         res => console.log(res));
  // }

  const fetchTweets =()=>{
    axios.post(`http://127.0.0.1:8000/tweets/`,{'data':props.tweetTable} )
      .then(res => {
        console.log(res);
        props.setAnalyze(res.data.analyzdata)
      }).catch((error)=>{
        console.log(error)
      })
  };

  useEffect(() => {
       fetchTweets()
  }, [props.setTweetTable, props.tweetTable])

  return (
    <Grid
      container
      direction="column"
      alignItems="center"
    >
      <Typography
        variant="h3"
        justifyContent="center"
        align="center"
        alignItems="center"
        sx={{ m: 2 }}
      >
        Insert Tweets
      </Typography>
      <TrialTable setTweetTable={props.setTweetTable} />
    </Grid>
  );
};

export default TweetsPage;
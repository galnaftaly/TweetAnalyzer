import React from "react";
import { Typography, Grid } from "@mui/material";
import TrialTable from "../TrialTable";

const TweetsPage = (props) => {
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
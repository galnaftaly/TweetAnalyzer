import React from "react";
import { Typography } from "@mui/material";
import TrialTable from "../TrialTable";

const TweetsPage = () => {
  return (
    <React.Fragment>
      <Typography
        variant="h3"
        justifyContent="center"
        align="center"
        sx={{ m: 2 }}
      >
        Insert Tweets
      </Typography>
      <TrialTable />
    </React.Fragment>
  );
};

export default TweetsPage;
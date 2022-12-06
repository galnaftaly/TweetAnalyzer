import React from 'react';
import { Typography } from '@mui/material';
import OurPieChart from '../PieChart';
import CustomizedTables from '../Table';

const DashboardPage = () => {
  return (
    <React.Fragment>
      <Typography
        variant="h3"
        justifyContent="center"
        align="center"
        sx={{ m: 2 }}
      >
        Dashboard
      </Typography>
      <OurPieChart />
      <CustomizedTables />
    </React.Fragment>
  );
};

export default DashboardPage;

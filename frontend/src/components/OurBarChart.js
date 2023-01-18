import React from 'react';
import {
  BarChart,
  Bar,
  Cell,
  Tooltip,
  XAxis,
  YAxis,
  CartesianGrid,
} from 'recharts';
import { Card, CardContent } from '@mui/material';

const COLORS = ['#FFBB28', '#FF8042', '#AF19FF', '#8884d8', '#82ca9d'];

function getBarData(props) {
  var sumFakeNews = 0;
  var sumTrueNews = 0;
  var countFakeNews = 0;
  var countTrueNews = 0;
  props.tweets.forEach((tweet) => {
    if (tweet.label === 1) {
      sumFakeNews += tweet.accuracy;
      countFakeNews += 1;
    } else if (tweet.label === 0) {
      sumTrueNews += tweet.accuracy;
      countTrueNews += 1;
    }
  });
  var barData = [
    { name: props.classes[0], value: sumTrueNews / countTrueNews },
    { name: props.classes[1], value: sumFakeNews / countFakeNews },
  ];
  return barData;
}

const OurBarChart = (props) => {
  const barData = getBarData(props);

  return (
    <Card>
      <CardContent>
        <BarChart width={500} height={300} data={barData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="name" style={{ fontFamily: 'Roboto' }} />
          <YAxis />
          <Tooltip contentStyle={{ fontFamily: 'Roboto' }} />
          <Bar dataKey="value" fill="#00a0fc" align="center">
            {barData.map((entry, index) => (
              <Cell
                key={`cell-${index}`}
                fill={COLORS[index % COLORS.length]}
              />
            ))}
          </Bar>
        </BarChart>
      </CardContent>
    </Card>
  );
};
export default OurBarChart;

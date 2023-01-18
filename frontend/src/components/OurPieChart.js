import React from 'react';
import { PieChart, Pie, Cell, Tooltip, Legend } from 'recharts';
import { Card, CardContent } from '@mui/material';

const COLORS = ['#8884d8', '#82ca9d', '#FFBB28', '#FF8042', '#AF19FF'];
function getPieData(props) {
  var countFakeNews = 0;
  var countTrueNews = 0;
  props.tweets.forEach((tweet) => {
    if (tweet.label === 1) countFakeNews += 1;
    else if (tweet.label === 0) countTrueNews += 1;
  });
  var pieData = [
    { name: props.classes[0], count: countTrueNews },
    { name: props.classes[1], count: countFakeNews },
  ];
  return pieData;
}

const OurPieChart = (props) => {
  const pieData = getPieData(props);

  return (
    <Card>
      <CardContent>
        <PieChart width={430} height={300}>
          <Pie
            data={pieData}
            color="#000000"
            dataKey="count"
            nameKey="name"
            cx="50%"
            cy="50%"
            outerRadius={120}
            fill="#8884d8"
          >
            {pieData.map((entry, index) => (
              <Cell
                key={`cell-${index}`}
                fill={COLORS[index % COLORS.length]}
              />
            ))}
          </Pie>
          <Tooltip contentStyle={{ fontFamily: 'Roboto' }} />
          <Legend wrapperStyle={{ fontFamily: 'Roboto' }} />
        </PieChart>
      </CardContent>
    </Card>
  );
};
export default OurPieChart;

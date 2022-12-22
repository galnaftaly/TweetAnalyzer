import React from 'react';
import { PieChart, Pie, Cell, Tooltip, Legend } from 'recharts';
import { Card, CardContent } from '@mui/material';

const COLORS = ['#8884d8', '#82ca9d', '#FFBB28', '#FF8042', '#AF19FF'];
function getPieData(tweets) {
  var countFakeNews = 0;
  var countTrueNews = 0;
  tweets.forEach((tweet) => {
    if (tweet.subject === 'Fake News') countFakeNews += 1;
    else if (tweet.subject === 'True News') countTrueNews += 1;
  });
  var pieData = [
    { name: 'Fake News', count: countFakeNews },
    { name: 'True News', count: countTrueNews },
  ];
  return pieData;
}

const OurPieChart = (props) => {
  const pieData = getPieData(props.tweets);

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

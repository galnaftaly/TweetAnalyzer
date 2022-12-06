import React from 'react';
import { PieChart, Pie, Cell, Tooltip, Legend } from 'recharts';
import { Box } from '@mui/material';

const COLORS = ['#8884d8', '#82ca9d', '#FFBB28', '#FF8042', '#AF19FF'];
const pieData = [
  {
    name: 'Apple',
    value: 54.85,
  },
  {
    name: 'Samsung',
    value: 47.91,
  },
  {
    name: 'Redmi',
    value: 16.85,
  },
  {
    name: 'One Plus',
    value: 16.14,
  },
  {
    name: 'Others',
    value: 10.25,
  },
];
function CustomTooltip(active, payload, label) {
  return (
    <div
      className="custom-tooltip"
      style={{
        backgroundColor: '#ffff',
        padding: '5px',
        border: '1px solid #cccc',
      }}
    >
      <label>{`${payload[0].name} : ${payload[0].value}%`}</label>
    </div>
  );
}

const OurPieChart = () => {
  return (
    <Box justifyContent="center" align="center">
      <PieChart width={730} height={300}>
        <Pie
          data={pieData}
          color="#000000"
          dataKey="value"
          nameKey="name"
          cx="50%"
          cy="50%"
          outerRadius={120}
          fill="#8884d8"
        >
          {pieData.map((entry, index) => (
            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
          ))}
        </Pie>
        <Tooltip element={<CustomTooltip />} />
        <Legend />
      </PieChart>
    </Box>
  );
};
export default OurPieChart;
//backgroundColor: ['#003f5c', '#58508d', '#bc5090', '#ff6361', '#ffa600'],

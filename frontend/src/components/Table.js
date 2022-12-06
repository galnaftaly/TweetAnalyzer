import * as React from 'react';
import { styled } from '@mui/material/styles';
import { Table, Card, Container, Grid } from '@mui/material';
import TableBody from '@mui/material/TableBody';
import TableCell, { tableCellClasses } from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Paper from '@mui/material/Paper';

const StyledTableCell = styled(TableCell)(({ theme }) => ({
  [`&.${tableCellClasses.head}`]: {
    backgroundColor: 'grey',
    color: theme.palette.common.white,
  },
  [`&.${tableCellClasses.body}`]: {
    fontSize: 14,
  },
}));

const StyledTableRow = styled(TableRow)(({ theme }) => ({
  '&:nth-of-type(odd)': {
    backgroundColor: theme.palette.action.hover,
  },
  // hide last border
  '&:last-child td, &:last-child th': {
    border: 0,
  },
}));

function createData(tweet, className, accuracy) {
  return { tweet, className, accuracy };
}

const dummy_rows = [
  createData('Tweet', 'Fake News', 59.1),
  createData('Tweet', 'Fake News', 68.4),
  createData('Tweet', 'True News', 75.0),
  createData('Tweet', 'Fake News', 89.7),
  createData('Tweet', 'True News', 78.4),
];

const CustomizedTables = () => {
  return (
    <Container
      spacing={0}
      direction="column"
      alignItems="center"
      justifyContent="center"
      style={{ minHeight: '100vh' }}
    >
      <TableContainer component={Paper}>
        <Table sx={{ minWidth: 700 }} aria-label="customized table">
          <TableHead sx={{ display: 'table-header-group' }}>
            <TableRow>
              <StyledTableCell align="right" sx={{ fontSize: 'h5.fontSize' }}>
                Tweet
              </StyledTableCell>
              <StyledTableCell align="right" sx={{ fontSize: 'h5.fontSize' }}>
                Class
              </StyledTableCell>
              <StyledTableCell align="right" sx={{ fontSize: 'h5.fontSize' }}>
                Accuracy
              </StyledTableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {dummy_rows.map((row, index) => (
              <TableRow key={index}>
                <StyledTableCell align="right">
                  {`${row.tweet} #${index + 1}`}
                </StyledTableCell>
                <StyledTableCell align="right">{row.className}</StyledTableCell>
                <StyledTableCell align="right">{row.accuracy}</StyledTableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    </Container>
  );
};

export default CustomizedTables;

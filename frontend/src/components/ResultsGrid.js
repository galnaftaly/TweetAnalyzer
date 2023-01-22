import React, { useState } from 'react';
import { Box, Alert, Typography } from '@mui/material';
import { DataGrid, GridToolbar, useGridApiRef } from '@mui/x-data-grid';
import ClickAwayListener from '@mui/base/ClickAwayListener';

const ResultsGrid = (props) => {
  const styles = {
    box: (theme) => ({
      height: 500,
      m: '2em',
      minWidth: 950,
      '& .super-app-theme--header': {
        backgroundColor: theme.palette.common.blue,
        color: 'white',
        fontSize: 20,
      },
    }),
  };

  const columns = [
    {
      field: 'id',
      headerName: 'Text',
      type: 'str',
      width: 300,
      editable: false,
      headerClassName: 'super-app-theme--header',
    },
    {
      field: 'label',
      headerName: 'Classification',
      type: 'str',
      width: 350,
      editable: false,
      headerClassName: 'super-app-theme--header',
    },
    {
      field: 'accuracy',
      headerName: 'Accuracy',
      type: 'float',
      width: 300,
      editable: false,
      headerClassName: 'super-app-theme--header',
    },
  ];

  const apiRef = useGridApiRef();
  const [selectedRow, setSelectedRow] = useState({});
  const [expandRow, setExpandRow] = useState(false);

  const handleRowClick = (params) => {
    setExpandRow(true);
    setSelectedRow(params.row);
  };

  const ExpandableRowContent = () => {
    return (
      <Typography variant="body1" align="left" sx={{ fontSize: 16 }}>
        Text {selectedRow.id} Content:
        <br />"{selectedRow.text}"
      </Typography>
    );
  };

  let rows = props.tweets.map(tweet => {
    let newLabel = tweet.label
    if (tweet.label === 0) newLabel = props.classes[0]
    else if (tweet.label === 1) newLabel = props.classes[1]
    return {...tweet, label: newLabel}
  })

  return (
    <ClickAwayListener onClickAway={() => setExpandRow(false)}>
      <Box align="center" sx={styles.box}>
        <DataGrid
          sx={{
            boxShadow: 2,
            border: 2,
            borderColor: 'primary.light',
            fontSize: 16,
          }}
          rows={rows}
          columns={columns}
          apiRef={apiRef}
          disableSelectionOnClick
          components={{ Toolbar: GridToolbar }}
          onRowClick={handleRowClick}
        />
        {expandRow && (
          <Alert severity="info">
            <ExpandableRowContent />
          </Alert>
        )}
      </Box>
    </ClickAwayListener>
  );
};

export default ResultsGrid;

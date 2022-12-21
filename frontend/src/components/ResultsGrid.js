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
      headerName: 'Tweet',
      type: 'int',
      width: 300,
      editable: false,
      headerClassName: 'super-app-theme--header',
    },
    {
      field: 'class',
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
        Tweet {selectedRow.id} Content:
        <br />"{selectedRow.content}"
      </Typography>
    );
  };

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
          rows={props.tweets}
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

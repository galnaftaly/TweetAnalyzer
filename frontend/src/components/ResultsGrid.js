import React from 'react';
import { Box, Tooltip } from '@mui/material';
import { DataGrid, GridToolbar, useGridApiRef } from '@mui/x-data-grid';

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
      renderCell: (params) => (
        <Tooltip title={params.value}>
          <span>{params.value}</span>
        </Tooltip>
      ),
    },
    {
      field: 'subject',
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

  return (
    <Box align="center" sx={styles.box}>
      <DataGrid
        sx={{ fontSize: 16 }}
        rows={props.tweets}
        columns={columns}
        apiRef={apiRef}
        disableSelectionOnClick
        components={{ Toolbar: GridToolbar }}
        experimentalFeatures={{ newEditingApi: true }}
      />
    </Box>
  );
};

export default ResultsGrid;

import React from 'react';
import { DialogTitle, Dialog, Box } from '@mui/material';
import model from '../assests/bgsrdmodel.png';

const ModelDialog = (props) => {
  const { open } = props;

  return (
    <Dialog
      maxWidth="lg"
      fullWidth={true}
      onClose={() => props.onClose()}
      open={open}
    >
      <DialogTitle align="center" variant="h6">
        BGSRD Model Architecture
      </DialogTitle>
      <Box
        component="img"
        alt="model image"
        src={model}
        sx={{ width: '100%', height: '100%' }}
      />
    </Dialog>
  );
};

export default ModelDialog;

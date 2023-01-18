import React, { useState } from 'react';
import {
  Typography,
  Grid,
  Box,
  InputLabel,
  MenuItem,
  FormControl,
  Select,
  Button,
} from '@mui/material';

const ChooseTask = (props) => {
  const [select, setSelect] = useState([]);
  const styles = {
    mainContainer: {
      my: '5em',
    },
    button: (theme) => ({
      height: 50,
      width: 200,
      fontSize: 20,
      m: '25px',
      borderRadius: 3,
      '&:hover': {
        backgroundColor: theme.palette.primary.light,
      },
    }),
  };

  const handleChange = (event) => {
    setSelect(event.target.value);
  };

  const sendingTaskChoose = (event) => {
    event.preventDefault();

    console.log(select);
    props.setTask(select);
    props.setDataset(select);
  };
  return (
    <div>
      <Box sx={{ minWidth: 700, my: 5 }}>
        <form onSubmit={sendingTaskChoose}>
          <Grid
            container
            spacing={3}
            direction="column"
            alignItems="center"
            justifyContent="center"
            sx={{ my: 7 }}
          >
            <Grid item>
              <Typography
                variant="h4"
                justifyContent="center"
                align="center"
                sx={{ mb: 1 }}
              >
                Choose which task you would like to explore:
              </Typography>
            </Grid>

            <Grid item>
              <Select
                labelId="demo-simple-select-label"
                id="demo-simple-select"
                width="1000px"
                variant="outlined"
                value={select}
                label="Task"
                onChange={handleChange}
                required
                sx={{ minWidth: 400 }}
              >
                <MenuItem value={'twitter'}>
                  Twitter - classify tweet - Fake/True news
                </MenuItem>
                <MenuItem value={'MR'}>
                  MR - classify movie review - Negative/Positive opinion
                </MenuItem>
                <MenuItem value={'shakespeare'}>
                  Shakespeare - classify book - Autorship Fake/Original
                </MenuItem>
              </Select>
            </Grid>
            <Grid item>
              <Button
                type="submit"
                size="large"
                variant="contained"
                sx={styles.button}
              >
                Continue
              </Button>
            </Grid>
          </Grid>
        </form>
      </Box>
      {/* <Box marginTop={3}  style={{justifyContent:'center'}} > */}

      {/* </Box> */}
    </div>
  );
};
export default ChooseTask;

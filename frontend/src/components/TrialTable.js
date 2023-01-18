import React, { useState, Fragment } from 'react';
import {
  TextField,
  Typography,
  CircularProgress,
  Alert,
  TableContainer,
  Button,
  Table,
  TableBody,
  TableHead,
  TableRow,
  Box,
  Container,
  Paper,
} from '@mui/material';
import { nanoid } from 'nanoid';
import ReadOnlyRow from './ReadOnlyRow';
import EditableRow from './EditableRow';
import { styled } from '@mui/material/styles';
import ContentPasteGoIcon from '@mui/icons-material/ContentPasteGo';
import AddBoxIcon from '@mui/icons-material/AddBox';
import TableCell, { tableCellClasses } from '@mui/material/TableCell';

const StyledTableCell = styled(TableCell)(({ theme }) => ({
  [`&.${tableCellClasses.head}`]: {
    backgroundColor: theme.palette.common.blue,
    color: theme.palette.common.white,
  },
  [`&.${tableCellClasses.body}`]: {
    fontSize: 14,
  },
}));

const TrialTable = (props) => {
  const [addFormData, setAddFormData] = useState({
    text: '',
  });
  const [value, setValue] = useState('');

  const [editFormData, setEditFormData] = useState({
    text: '',
  });
  const [alert, setAlert] = useState(false);

  const [editTweetId, setEditTweetId] = useState(null);

  const handleAddFormChange = (event) => {
    event.preventDefault();
    setValue(event.target.value);
    const fieldName = event.target.getAttribute('name');
    const fieldValue = event.target.value;

    const newFormData = { ...addFormData };
    newFormData[fieldName] = fieldValue;

    setAddFormData(newFormData);
  };

  const handleEditFormChange = (event) => {
    event.preventDefault();

    const fieldName = event.target.getAttribute('name');
    const fieldValue = event.target.value;

    const newFormData = { ...editFormData };
    newFormData[fieldName] = fieldValue;

    setEditFormData(newFormData);
  };

  const handleAddFormSubmit = (event) => {
    event.preventDefault();
    setValue('');

    const newTweet = {
      id: nanoid(),
      text: addFormData.text,
    };

    const newTweets = [...props.tweets, newTweet];
    props.setTweets(newTweets);
    setAlert(false);
  };

  const handleEditFormSubmit = (event) => {
    event.preventDefault();

    const editedTweet = {
      id: editTweetId,
      text: editFormData.text,
    };

    const newTweets = [...props.tweets];

    const index = props.tweets.findIndex((tweet) => tweet.id === editTweetId);

    newTweets[index] = editedTweet;

    props.setTweets(newTweets);
    setEditTweetId(null);
  };

  const handleEditClick = (event, tweet) => {
    event.preventDefault();
    setEditTweetId(tweet.id);

    const formValues = {
      text: tweet.text,
    };

    setEditFormData(formValues);
  };

  const handleCancelClick = () => {
    setEditTweetId(null);
  };

  const handleDeleteClick = (tweetId) => {
    const newTweets = [...props.tweets];
    const index = props.tweets.findIndex((tweet) => tweet.id === tweetId);
    newTweets.splice(index, 1);
    props.setTweets(newTweets);
  };

  const sendingTweetsTable = () => {
    if (props.tweets.length === 0) {
      setAlert(true);
    } else {
      props.setFetch(true);
      props.setTweetTable(props.tweets);
    }
  };

  return (
    <Box
      spacing={1}
      direction="column"
      align="center"
      //sx={{ minHeight: '100vh', m: '2' }}
    >
      <form onSubmit={handleEditFormSubmit}>
        <TableContainer component={Paper} sx={{ maxHeight: 600, width: 1000 }}>
          <Table
            sx={{
              width: 'max-content',
              minWidth: 1000,
              maxWidth: 1000,
              borderCollapse: 'collapse',
            }}
            stickyHeader={true}
            aria-label="customized table"
          >
            <TableHead sx={{ display: 'table-header-group' }}>
              <TableRow>
                <StyledTableCell
                  align="left"
                  sx={{
                    fontSize: 'h5.fontSize',
                    border: 1,
                    borderColor: 'primary.main',
                  }}
                >
                  {props.task === 'twitter'
                    ? 'Tweets Content'
                    : 'Reviews Content'}
                </StyledTableCell>
                <StyledTableCell
                  align="left"
                  width="230"
                  sx={{
                    border: 1,
                    borderColor: 'primary.main',
                    fontSize: 'h5.fontSize',
                  }}
                >
                  Actions
                </StyledTableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {props.tweets.map((tweet, index) => (
                <Fragment>
                  {editTweetId === tweet.id ? (
                    <EditableRow
                      editFormData={editFormData}
                      handleEditFormChange={handleEditFormChange}
                      handleCancelClick={handleCancelClick}
                      key={`${tweet.id}${index}`}
                    />
                  ) : (
                    <ReadOnlyRow
                      key={`${index}`}
                      tweet={tweet}
                      handleEditClick={handleEditClick}
                      handleDeleteClick={handleDeleteClick}
                    />
                  )}
                </Fragment>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </form>

      <Container
        spacing={1}
        direction="column"
        justifyContent="center"
        //style={{ minHeight: '100vh' }}
      >
        <Typography
          variant="h4"
          justifyContent="center"
          align="center"
          sx={{ m: 3 }}
        >
          {props.task === 'twitter' ? 'Add a new Tweet:' : 'Add a new Review:'}
        </Typography>
        <form onSubmit={handleAddFormSubmit}>
          <TextField
            align="center"
            id="standard-basic"
            autoFocus={true}
            margin="normal"
            variant="standard"
            type="text"
            multiline={true}
            name="text"
            required
            sx={{ width: 700 }}
            label={
              props.task === 'twitter'
                ? 'Enter a Text from tweet...'
                : 'Enter a Text from Review...'
            }
            // label="Enter a Text from tweet..."
            onChange={handleAddFormChange}
            value={value}
          />

          <Button endIcon={<AddBoxIcon />} type="submit">
            Add
          </Button>
          <Box sx={{ m: 2 }}>
            <Button
              size="large"
              variant="contained"
              color="success"
              onClick={sendingTweetsTable}
              endIcon={<ContentPasteGoIcon />}
              sx={{ m: 2 }}
            >
              Analyze
            </Button>
            {alert && (
              <Alert severity="error">
                <Typography sx={{ fontSize: 16 }}>
                  You must insert text to analyze
                </Typography>
              </Alert>
            )}
          </Box>
        </form>
      </Container>
    </Box>
  );
};

export default TrialTable;

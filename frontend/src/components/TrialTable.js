import React, { useState, Fragment } from "react";
import { TextField, Typography } from "@mui/material";
import { nanoid } from "nanoid";
import ReadOnlyRow from "./ReadOnlyRow";
import EditableRow from "./EditableRow";
import { styled } from "@mui/material/styles";
import Paper from '@mui/material/Paper';
import ContentPasteGoIcon from '@mui/icons-material/ContentPasteGo';
import {
  TableContainer,
  Button,
  Table,
  TableBody,
  TableHead,
  TableRow,
  Box
} from "@mui/material";
import AddBoxIcon from "@mui/icons-material/AddBox";
import TableCell, { tableCellClasses } from "@mui/material/TableCell";
import { Container } from "@mui/system";

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
  const [tweets, setTweets] = useState([]);
  const [addFormData, setAddFormData] = useState({
    tweetText: "",
  });

  const [editFormData, setEditFormData] = useState({
    tweetText: "",
  });

  const [editTweetId, setEditTweetId] = useState(null);

  const handleAddFormChange = (event) => {
    event.preventDefault();

    const fieldName = event.target.getAttribute("name");
    const fieldValue = event.target.value;

    const newFormData = { ...addFormData };
    newFormData[fieldName] = fieldValue;

    setAddFormData(newFormData);
  };

  const handleEditFormChange = (event) => {
    event.preventDefault();

    const fieldName = event.target.getAttribute("name");
    const fieldValue = event.target.value;

    const newFormData = { ...editFormData };
    newFormData[fieldName] = fieldValue;

    setEditFormData(newFormData);
  };

  const handleAddFormSubmit = (event) => {
    event.preventDefault();

    const newTweet = {
      id: nanoid(),
      tweetText: addFormData.tweetText,
    };

    const newTweets = [...tweets, newTweet];
    setTweets(newTweets);
  };

  const handleEditFormSubmit = (event) => {
    event.preventDefault();

    const editedTweet = {
      id: editTweetId,
      tweetText: editFormData.tweetText,
    };

    const newTweets = [...tweets];

    const index = tweets.findIndex((tweet) => tweet.id === editTweetId);

    newTweets[index] = editedTweet;

    setTweets(newTweets);
    setEditTweetId(null);
  };

  const handleEditClick = (event, tweet) => {
    event.preventDefault();
    setEditTweetId(tweet.id);

    const formValues = {
      tweetText: tweet.tweetText,
    };

    setEditFormData(formValues);
  };

  const handleCancelClick = () => {
    setEditTweetId(null);
  };

  const handleDeleteClick = (tweetId) => {
    const newTweets = [...tweets];

    const index = tweets.findIndex((tweet) => tweet.id === tweetId);

    newTweets.splice(index, 1);

    setTweets(newTweets);
  };
  const sendingTweetsTable=()=>{
    props.setTweetTable(tweets)

  };

  return (
    <Box
      spacing={1}
      direction="column"
      align="center"
      
      sx={{minHeight: "100vh", m:"2"}}
  
    >
      < form onSubmit={handleEditFormSubmit}>
        <TableContainer component={Paper} sx={{  maxHeight: 600 , width:1000   }}>
          <Table sx={{width:"max-content" ,minWidth: 1000, maxWidth:1000 , borderCollapse:"collapse"  }}   stickyHeader={true} aria-label="customized table">
            <TableHead sx={{ display: 'table-header-group' }}>
              <TableRow >
                <StyledTableCell align="left" sx={{ fontSize: "h5.fontSize" , border:1, borderColor: 'primary.main' }}>
                  Tweets Content
                </StyledTableCell>
                <StyledTableCell
                  align="left"
                  width="230"
                  sx={{border:1,borderColor: 'primary.main' , fontSize: "h5.fontSize" }}
                >
                  Actions
                </StyledTableCell>
              </TableRow>
            </TableHead>
            <TableBody >
              {tweets.map((tweet) => (
                <Fragment>
                  {editTweetId === tweet.id ? (
                    <EditableRow
                   
                      editFormData={editFormData}
                      handleEditFormChange={handleEditFormChange}
                      handleCancelClick={handleCancelClick}
                    />
                  ) : (
                    <ReadOnlyRow
                   
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
      </ form>

      
      <Container
      spacing={1}
      direction="column"
      justifyContent="center"
      style={{ minHeight: "100vh" }}
    >
      <Typography
        variant="h4"
        justifyContent="center"
        align="center"
        sx={{ m: 3 }}
      >
        Add a new Tweet:
      </Typography>
      < form onSubmit={handleAddFormSubmit}>
        <TextField
          align="center"
          id="standard-basic"
          autoFocus={true}
          margin="normal"
          variant="standard"
          type="text"
          multiline={true}
          name="tweetText"
          required
          sx={{width:700}}
          label="Enter a Text from tweet..."
          onChange={handleAddFormChange}
        />

        <Button  endIcon={<AddBoxIcon />} type="submit">
          Add
        </Button>
        <Box marginTop={5}>
      <Button size="large" variant="contained" color="success" onClick={sendingTweetsTable}  endIcon={<ContentPasteGoIcon />} >
          Analyze
        </Button>
        </Box>
      </ form>
      </Container>
    </Box>
  );
};

export default TrialTable;

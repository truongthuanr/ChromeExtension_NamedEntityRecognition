$(function(){

    
    $('#keywordsubmit').click(function(){
		
		var search_topic = $('#Textarea').val();
		
				
		if (search_topic){
                chrome.runtime.sendMessage(
					{topic: search_topic},
					function(response) {
						result = response.farewell;
						alert(result.summary);
						
						var notifOptions = {
                        type: "basic",
                        iconUrl: "icon48.png",
                        title: "Result for NER Tag",
                        message: result.summary
						};
						
						chrome.notifications.create('WikiNotif', notifOptions);
						
					});
		}
			
			
		$('#Textarea').val('');
		
    });
});